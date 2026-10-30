"""tests for the desktop app helpers, screen metrics, and checkpoint loading."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from air.model import AirModel
from desktop.app import build_report_text, find_checkpoint
from model.spectrogram_cnn import SpectrogramCNN
from predict import LABEL_NAMES, load_model, state_dict_from_checkpoint
from screen.anatomy_probe import anatomy_metrics, load_xray, observations


def _synthetic_xray() -> np.ndarray:
    """fake chest x-ray: bright lung fields, dark mediastinum, darker base."""
    img = np.zeros((256, 256), dtype=np.float32)
    img[:, :] = 0.35  # soft tissue
    img[40:200, 20:110] = 0.6  # right field bright
    img[40:200, 150:235] = 0.55  # left field slightly darker
    img[:, 110:150] = 0.1  # mediastinum shadow
    return img


def test_state_dict_from_checkpoint_train_format(tmp_path: Path):
    model = SpectrogramCNN(n_mels=64, n_classes=4)
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "epochs": 3}, ckpt)
    loaded = state_dict_from_checkpoint(ckpt)
    assert set(loaded.keys()) == set(model.state_dict().keys())


def test_state_dict_from_checkpoint_bare(tmp_path: Path):
    model = SpectrogramCNN(n_mels=64, n_classes=4)
    ckpt = tmp_path / "bare.pt"
    torch.save(model.state_dict(), ckpt)
    loaded = state_dict_from_checkpoint(ckpt)
    assert set(loaded.keys()) == set(model.state_dict().keys())


def test_load_model_roundtrip(tmp_path: Path):
    model = SpectrogramCNN(n_mels=64, n_classes=4)
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    restored = load_model(ckpt, "cpu")
    x = torch.randn(1, 1, 64, 100)
    assert restored(x).shape == (1, 4)


def test_find_checkpoint_env_precedence(tmp_path: Path, monkeypatch):
    target = tmp_path / "custom.pt"
    target.touch()
    monkeypatch.setenv("ASTHMA_CHECKPOINT", str(target))
    assert find_checkpoint() == target


def test_build_report_text_empty():
    text = build_report_text(None, None)
    assert "asthma risk report" in text
    assert "no recording analysed" in text
    assert "no forecast fetched" in text


def test_build_report_text_high_risk_combo():
    breath = {"prediction": "both", "probabilities": {name: 0.25 for name in LABEL_NAMES}}
    breath["probabilities"]["both"] = 0.6
    air = {"forecast_pm25": 90.0, "risk_band": AirModel.risk_band(90.0)}
    text = build_report_text(breath, air, "Delhi")
    assert "high risk" in text
    assert "rescue inhaler" in text
    assert "research tool" in text


def test_anatomy_metrics_synthetic(tmp_path: Path):
    png = tmp_path / "cxr.png"
    Image.fromarray((_synthetic_xray() * 255).astype(np.uint8)).save(png)
    img = load_xray(png)
    metrics = anatomy_metrics(img)
    assert set(metrics) == {"transparency", "asymmetry", "hilar_texture", "lower_dominance"}
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, name
    notes = observations(metrics)
    assert notes and all(isinstance(n, str) for n in notes)


def test_save_pdf(tmp_path: Path):
    from desktop.pdf_report import save_pdf

    target = save_pdf("asthma risk report\nline two", tmp_path / "report.pdf")
    data = target.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 200
