"""fetch weather and particulate matter for a city.

weather comes from open meteo (no key). particulate matter comes from open aq
when OPENAQ_KEY is set, otherwise the run records a warning and trains on
weather only.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import requests


OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    params = {
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
    key = os.environ.get("OPENAQ_KEY")
    if not key:
        print("warning: OPENAQ_KEY not set, skipping particulate matter")
        return None
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = "https://api.openaq.org/v2/measurements"
    headers = {"X-API-Key": key}
    params = {
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
            records.append(
                {"time": measurement["lastUpdated"], "pm25": measurement["value"]}
            )
    if not records:
        return None
    series = pd.DataFrame(records)
    series["time"] = pd.to_datetime(series["time"])
    series = series.set_index("time").resample("D").mean()["pm25"]
    return series.dropna()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    weather = fetch_weather(args.lat, args.lon)
    pm = fetch_pm25(args.lat, args.lon)
    if pm is not None:
        weather = weather.join(pm, how="inner")
    print(weather.tail(7))
    weather.to_csv(f"air_{args.city}.csv")
    print(f"wrote air_{args.city}.csv")


if __name__ == "__main__":
    main()
