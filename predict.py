"""infer the breath class of a single recording with a trained checkpoint.

usage:
    python predict.py --checkpoint out/model.pt --wav path/to/file.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from model.spectrogram_cnn import SpectrogramCNN

LABEL_NAMES = ["normal", "crackle", "wheeze", "both"]
TARGET_SR = 16000
N_MELS = 64


def state_dict_from_checkpoint(path: str | Path) -> dict:
    """accept both train.py checkpoints ({model_state_dict, ...}) and bare state dicts."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "model_state_dict" in data:
        return data["model_state_dict"]
    if (
        isinstance(data, dict)
        and "model" in data
        and not any(k.startswith(("conv", "fc", "blocks")) for k in data)
    ):
        return data["model"]
    return data


def load_model(checkpoint: str | Path, device: torch.device | str) -> SpectrogramCNN:
    model = SpectrogramCNN()
    model.load_state_dict(state_dict_from_checkpoint(checkpoint))
    model.to(device)
    model.eval()
    return model


def load_wav(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def spec_from_wav(waveform: torch.Tensor) -> torch.Tensor:
    mel = MelSpectrogram(sample_rate=TARGET_SR, n_mels=N_MELS)
    db = AmplitudeToDB()
    spec = db(mel(waveform)).clamp(-80.0, 0.0)
    return (spec + 80.0) / 80.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to model.pt")
    parser.add_argument("--wav", required=True, help="path to a wav file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    model = load_model(args.checkpoint, device)
    model.eval()

    waveform = load_wav(Path(args.wav))
    spec = spec_from_wav(waveform).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=1)[0]

    pred = int(probs.argmax())
    print(f"prediction: {LABEL_NAMES[pred]}")
    for name, prob in zip(LABEL_NAMES, probs.tolist()):
        print(f"  {name:8s} {prob:.3f}")


if __name__ == "__main__":
    main()
