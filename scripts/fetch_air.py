"""fetch weather and particulate matter for a location.

weather comes from open meteo's archive api (no key). particulate matter comes
from open meteo's air quality api (also no key), falling back to open aq when
OPENAQ_KEY is set. the run records a warning and trains on weather only when
both fail.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_weather(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    end = datetime.now(UTC).date() - timedelta(days=1)  # archive lags a day
    start = end - timedelta(days=days)
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start),
        "end_date": str(end),
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
        "timezone": "UTC",
    }
    response = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").resample("D").mean().dropna()
    return df


def fetch_pm25(lat: float, lon: float, days: int = 30) -> pd.Series | None:
    """daily pm two point five from open meteo's air quality api (no key)."""
    try:
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5",
            "past_days": min(days, 92),  # the api caps past_days at 92
            "timezone": "UTC",
        }
        response = requests.get(OPEN_METEO_AIR_QUALITY, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()["hourly"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        daily = df.set_index("time").resample("D").mean()["pm2_5"].dropna()
        if daily.empty:
            print("warning: open meteo air quality returned no pm two point five")
        return daily.rename("pm25")
    except Exception as e:  # noqa: BLE001 - degrade to weather-only forecasting
        print(f"warning: open meteo air quality failed ({e})")
        return fetch_pm25_openaq(lat, lon, days)


def fetch_pm25_openaq(lat: float, lon: float, days: int = 30) -> pd.Series | None:
    """legacy open aq source, requires OPENAQ_KEY."""
    key = os.environ.get("OPENAQ_KEY")
    if not key:
        return None
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    url = "https://api.openaq.org/v2/measurements"
    headers = {"X-API-Key": key}
    params: dict[str, Any] = {
        "coordinates": f"{lat},{lon}",
        "parameter": "pm25",
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "limit": 10000,
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        print(f"warning: open aq returned {response.status_code}")
        return None
    results = response.json().get("results", [])
    if not results:
        return None
    records = []
    for item in results:
        for measurement in item.get("measurements", []):
            records.append({"time": measurement["lastUpdated"], "pm25": measurement["value"]})
    if not records:
        return None
    series = pd.DataFrame(records)
    series["time"] = pd.to_datetime(series["time"])
    series = series.set_index("time").resample("D").mean()["pm25"]
    return series.dropna()


def fetch_air_quality(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """weather + pm joined on date; keeps weather only when pm is unavailable."""
    weather = fetch_weather(lat, lon, days)
    pm = fetch_pm25(lat, lon, days)
    if pm is not None:
        weather = weather.join(pm, how="inner")
    if "pm25" not in weather.columns:
        raise ValueError("pm two point five unavailable for this location right now")
    return weather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    frame = fetch_air_quality(args.lat, args.lon, 30)
    print(frame.tail(7))
    frame.to_csv(f"air_{args.city}.csv")
    print(f"wrote air_{args.city}.csv")


if __name__ == "__main__":
    main()
