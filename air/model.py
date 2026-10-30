"""next day pm two point five forecast and risk bands.

input: daily frame with weather columns and optional pm two point five.
builds lag features from the past week and fits a random forest regressor.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

LAGS = 7


def add_lags(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    for lag in range(1, LAGS + 1):
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    return out


class AirModel:
    def __init__(self, target: str = "pm25"):
        self.target = target
        self.model = RandomForestRegressor(n_estimators=200, random_state=0)
        self.feature_cols: list[str] = []

    def fit(self, frame: pd.DataFrame) -> None:
        data = add_lags(frame, self.target).dropna()
        if self.target not in data.columns:
            raise ValueError(f"{self.target} column missing")
        self.feature_cols = [c for c in data.columns if c != self.target]
        self.model.fit(data[self.feature_cols], data[self.target])

    def forecast_next_day(self, frame: pd.DataFrame) -> float:
        data = add_lags(frame, self.target).dropna().tail(1)
        return float(self.model.predict(data[self.feature_cols])[0])

    @staticmethod
    def risk_band(pm25: float) -> str:
        if pm25 < 35:
            return "low"
        if pm25 < 75:
            return "medium"
        return "high"


def run(csv_path: str) -> None:
    frame = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    if "pm25" not in frame.columns:
        raise ValueError("pm two point five column required for forecasting")
    model = AirModel()
    model.fit(frame)
    forecast = model.forecast_next_day(frame)
    print(f"forecast pm two point five: {forecast:.1f}")
    print(f"risk band: {model.risk_band(forecast)}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
