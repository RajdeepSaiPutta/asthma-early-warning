"""Tests for asthma-early-warning."""

import numpy as np
import torch
from pathlib import Path

from data.icbhi import (
    parse_annotation,
    recording_label,
    patient_id,
    list_recordings,
    cycle_label,
    Cycle,
)
from model.spectrogram_cnn import SpectrogramCNN
from air.model import AirModel


def test_cycle_label():
    """Test cycle label mapping."""
    assert cycle_label(Cycle(0, 1, 0, 0)) == 0  # normal
    assert cycle_label(Cycle(0, 1, 1, 0)) == 1  # crackle
    assert cycle_label(Cycle(0, 1, 0, 1)) == 2  # wheeze
    assert cycle_label(Cycle(0, 1, 1, 1)) == 3  # both


def test_recording_label():
    """Test recording label aggregation."""
    # Normal
    assert recording_label([Cycle(0, 1, 0, 0)]) == 0
    # Crackle
    assert recording_label([Cycle(0, 1, 1, 0)]) == 1
    # Wheeze
    assert recording_label([Cycle(0, 1, 0, 1)]) == 2
    # Both (single cycle with both)
    assert recording_label([Cycle(0, 1, 1, 1)]) == 3
    # Mixed - takes max of cycle labels (both=3 > wheeze=2 > crackle=1 > normal=0)
    # crackle(1) + wheeze(2) in separate cycles -> max(1,2) = 2 (wheeze)
    assert recording_label([Cycle(0, 1, 0, 0), Cycle(0, 1, 1, 0)]) == 1
    assert recording_label([Cycle(0, 1, 0, 0), Cycle(0, 1, 0, 1)]) == 2
    assert recording_label([Cycle(0, 1, 1, 0), Cycle(0, 1, 0, 1)]) == 2
    # Only returns 3 if a SINGLE cycle has both crackle AND wheeze
    assert recording_label([Cycle(0, 1, 1, 1)]) == 3


def test_patient_id():
    """Test patient ID extraction."""
    assert patient_id("101_1b1_Al_sc_Meditron") == "101"
    assert patient_id("102_2b1_Ar_sc_Litt3200") == "102"
    assert patient_id("103") == "103"


def test_spectrogram_cnn_forward():
    """Test model forward pass."""
    model = SpectrogramCNN(n_mels=64, n_classes=4)
    x = torch.randn(2, 1, 64, 100)  # batch, channels, n_mels, time
    logits = model(x)
    assert logits.shape == (2, 4)


def test_air_model_risk_bands():
    """Test air model risk band classification."""
    assert AirModel.risk_band(20) == "low"
    assert AirModel.risk_band(35) == "medium"
    assert AirModel.risk_band(50) == "medium"
    assert AirModel.risk_band(75) == "high"
    assert AirModel.risk_band(100) == "high"


def test_air_model_add_lags():
    """Test lag feature generation."""
    import pandas as pd
    from air.model import add_lags
    df = pd.DataFrame({"pm25": [10, 20, 30, 40, 50, 60, 70, 80]})
    result = add_lags(df, "pm25")
    assert "pm25_lag1" in result.columns
    assert "pm25_lag7" in result.columns
    # Check that lag1 is shifted correctly
    assert result["pm25_lag1"].iloc[1] == 10
    assert result["pm25_lag7"].iloc[7] == 10