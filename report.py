"""combine breath and air signals into a per-patient risk report.

usage:
    python report.py --breath out/model.pt --wav data/patient.wav \
        --air air_hyderabad.csv --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import load_wav, spec_from_wav, LABEL_NAMES, load_model  # noqa: E402
from model.spectrogram_cnn import SpectrogramCNN  # noqa: E402
from air.model import AirModel  # noqa: E402


def breath_prediction(checkpoint: str, wav: Path, device: str) -> dict:
    model = load_model(checkpoint, device)

    waveform = load_wav(wav)
    spec = spec_from_wav(waveform).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=1)[0]

    pred = int(probs.argmax())
    return {
        "prediction": LABEL_NAMES[pred],
        "probabilities": {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs.tolist())},
    }


def air_forecast(csv_path: str | None) -> dict | None:
    if csv_path is None:
        return None
    frame = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    if "pm25" not in frame.columns:
        return None
    model = AirModel()
    model.fit(frame)
    forecast = model.forecast_next_day(frame)
    return {
        "forecast_pm25": round(float(forecast), 2),
        "risk_band": AirModel.risk_band(forecast),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--breath", required=True, help="trained breath checkpoint")
    parser.add_argument("--wav", required=True, help="patient lung-sound wav")
    parser.add_argument("--air", default=None, help="air quality csv (optional)")
    parser.add_argument("--out", default="report.json", help="output report path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report = {
        "patient_wav": str(args.wav),
        "breath": breath_prediction(args.breath, Path(args.wav), args.device),
        "air": air_forecast(args.air),
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
