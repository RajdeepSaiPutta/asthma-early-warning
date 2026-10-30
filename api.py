"""FastAPI web API for the asthma early warning system.

Three panels:
- /breath: classify lung sound recordings
- /air: forecast next-day PM2.5 risk
- /screen: exploratory chest x-ray anatomy metrics
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from air.model import AirModel
from data.icbhi import list_recordings
from model.spectrogram_cnn import SpectrogramCNN
from predict import load_wav, spec_from_wav, LABEL_NAMES

app = FastAPI(
    title="Asthma Early Warning API",
    version="0.2.0",
    description="Three-panel asthma risk API: breath, air, screen",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (loaded on startup)
breath_model: Optional[SpectrogramCNN] = None
breath_device: Optional[torch.device] = None
air_model: Optional[AirModel] = None


class BreathResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class AirResponse(BaseModel):
    forecast_pm25: float
    risk_band: str


class ReportResponse(BaseModel):
    breath: BreathResponse
    air: Optional[AirResponse] = None


class ScreenResponse(BaseModel):
    exploratory: bool = True
    metrics: dict[str, float] | None = None
    observations: list[str] | None = None
    disclaimer: str = (
        "exploratory anatomy metrics from an unlabeled tool - discussion aids, not diagnosis"
    )


@app.on_event("startup")
async def load_models():
    global breath_model, breath_device, air_model

    breath_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_model = SpectrogramCNN().to(breath_device)
    breath_model.eval()

    # Load best checkpoint if available
    ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=breath_device)
        breath_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded breath model from {ckpt_path}")

    # Air model will be trained on demand
    air_model = AirModel()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "breath_model_loaded": breath_model is not None,
        "air_model_ready": air_model is not None,
    }


@app.post("/breath/classify", response_model=BreathResponse)
async def classify_breath(file: UploadFile = File(...)):
    """Classify a lung sound recording."""
    if breath_model is None:
        raise HTTPException(status_code=503, detail="Breath model not loaded")

    # Save uploaded file temporarily
    content = await file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        waveform = load_wav(Path(tmp_path))
        spec = spec_from_wav(waveform).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            logits = breath_model(spec)
            probs = torch.softmax(logits, dim=1)[0]

        pred = int(probs.argmax())
        return BreathResponse(
            prediction=LABEL_NAMES[pred],
            probabilities={name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs.tolist())},
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/air/forecast", response_model=AirResponse)
async def forecast_air(
    city: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    days: int = Form(30),
):
    """Fetch weather/PM2.5 and forecast next-day PM2.5 risk."""
    from scripts.fetch_air import fetch_weather, fetch_pm25

    weather = fetch_weather(lat, lon, days)
    pm = fetch_pm25(lat, lon, days)

    if pm is not None:
        weather = weather.join(pm, how="inner")

    if "pm25" not in weather.columns:
        raise HTTPException(status_code=400, detail="PM2.5 data unavailable for this location")

    model = AirModel()
    model.fit(weather)
    forecast = model.forecast_next_day(weather)

    return AirResponse(
        forecast_pm25=round(float(forecast), 1),
        risk_band=AirModel.risk_band(forecast),
    )


@app.post("/report", response_model=ReportResponse)
async def generate_report(
    wav_file: UploadFile = File(...),
    air_city: Optional[str] = Form(None),
    air_lat: Optional[float] = Form(None),
    air_lon: Optional[float] = Form(None),
):
    """Generate combined breath + air risk report."""
    # Breath classification
    if breath_model is None:
        raise HTTPException(status_code=503, detail="Breath model not loaded")

    content = await wav_file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        waveform = load_wav(Path(tmp_path))
        spec = spec_from_wav(waveform).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            logits = breath_model(spec)
            probs = torch.softmax(logits, dim=1)[0]

        pred = int(probs.argmax())
        breath_result = BreathResponse(
            prediction=LABEL_NAMES[pred],
            probabilities={name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs.tolist())},
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Air forecast (if coordinates provided)
    air_result = None
    if air_city and air_lat is not None and air_lon is not None:
        from scripts.fetch_air import fetch_weather, fetch_pm25
        weather = fetch_weather(air_lat, air_lon, 30)
        pm = fetch_pm25(air_lat, air_lon, 30)
        if pm is not None:
            weather = weather.join(pm, how="inner")
        if "pm25" in weather.columns:
            model = AirModel()
            model.fit(weather)
            forecast = model.forecast_next_day(weather)
            air_result = AirResponse(
                forecast_pm25=round(float(forecast), 1),
                risk_band=AirModel.risk_band(forecast),
            )

    return ReportResponse(breath=breath_result, air=air_result)


@app.post("/screen/analyze", response_model=ScreenResponse)
async def analyze_screen(file: UploadFile = File(...)):
    """Exploratory chest x-ray anatomy metrics (no trained risk classifier)."""
    from screen.anatomy_probe import anatomy_metrics, load_xray, observations

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        img = load_xray(tmp_path)
        metrics = anatomy_metrics(img)
        return ScreenResponse(metrics=metrics, observations=observations(metrics))
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)