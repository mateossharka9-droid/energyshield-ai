"""Synthetic Albania weather context for EnergyShield NTL.

This module creates a deterministic, realistic daily temperature dataset for
Albanian cities and attaches it to meter readings. It is designed for prototype
use when real utility/weather feeds are not available.

Production replacement:
- Replace generate_albania_weather_dataset() with Open-Meteo, Meteostat, NASA
  POWER, or national meteorological station data.
- Join by customer GIS coordinates, city/area, and date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ALBANIA_CITY_CLIMATE = {
    # city: base annual mean, seasonal amplitude, longitude, latitude
    "Tirana":  (16.2, 10.7, 19.8187, 41.3275),
    "Durres":  (16.7,  9.5, 19.4414, 41.3231),
    "Shkoder": (15.8, 10.1, 19.5033, 42.0693),
    "Vlore":   (17.4,  9.1, 19.4914, 40.4661),
    "Elbasan": (16.0, 11.2, 20.0822, 41.1125),
    "Fier":    (16.8, 10.0, 19.5561, 40.7239),
    "Korce":   (11.9, 12.0, 20.7808, 40.6186),
    "Kukes":   (11.6, 12.4, 20.4217, 42.0767),
}


def classify_weather(temp_mean: float, cold_threshold: float = 10.0, hot_threshold: float = 26.0) -> int:
    """Return -1 cold, 0 normal, 1 hot based on average daily temperature."""
    if pd.isna(temp_mean):
        return 0
    if temp_mean < cold_threshold:
        return -1
    if temp_mean > hot_threshold:
        return 1
    return 0


def generate_albania_weather_dataset(
    start_date: str = "2014-01-01",
    end_date: str = "2016-10-31",
    output_path: str | Path | None = None,
    cities: Iterable[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily synthetic Albania temperature/weather context.

    The output is deterministic for a seed and includes the engineered features
    used by the NTL model:
    - weather_class: -1 cold, 0 normal, 1 hot
    - weather_demand_pressure: 1 on cold/hot days, 0 on normal days
    - heating_degree_days and cooling_degree_days
    """
    rng = np.random.default_rng(seed)
    date_index = pd.date_range(start_date, end_date, freq="D")
    selected_cities = list(cities) if cities is not None else list(ALBANIA_CITY_CLIMATE.keys())
    records = []

    for city in selected_cities:
        base, amplitude, lon, lat = ALBANIA_CITY_CLIMATE.get(city, ALBANIA_CITY_CLIMATE["Tirana"])
        # Peak temperature around late July / early August.
        day_of_year = date_index.dayofyear.to_numpy()
        seasonal = amplitude * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
        weekly_noise = rng.normal(0, 1.6, size=len(date_index))
        synoptic_noise = pd.Series(rng.normal(0, 1.1, size=len(date_index))).rolling(5, min_periods=1).mean().to_numpy()
        temp_mean = base + seasonal + weekly_noise + synoptic_noise
        temp_min = temp_mean - rng.uniform(4.5, 8.0, size=len(date_index))
        temp_max = temp_mean + rng.uniform(5.0, 9.5, size=len(date_index))
        # More precipitation in autumn/winter, less in summer.
        wet_season_factor = 0.55 + 0.45 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
        rain_probability = np.clip(0.15 + 0.30 * wet_season_factor, 0.05, 0.62)
        precipitation = np.where(rng.random(len(date_index)) < rain_probability, rng.gamma(1.4, 6.0, len(date_index)), 0.0)

        for date, tmean, tmin, tmax, rain in zip(date_index, temp_mean, temp_min, temp_max, precipitation):
            wclass = classify_weather(float(tmean))
            hdd = max(18.0 - float(tmean), 0.0)
            cdd = max(float(tmean) - 24.0, 0.0)
            records.append({
                "date": date.date().isoformat(),
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "temp_mean": round(float(tmean), 2),
                "temp_min": round(float(tmin), 2),
                "temp_max": round(float(tmax), 2),
                "precipitation_mm": round(float(rain), 2),
                "heating_degree_days": round(float(hdd), 2),
                "cooling_degree_days": round(float(cdd), 2),
                "weather_class": int(wclass),
                "is_cold": int(wclass == -1),
                "is_hot": int(wclass == 1),
                "weather_demand_pressure": int(abs(wclass)),
            })

    weather = pd.DataFrame(records)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weather.to_csv(output_path, index=False)
    return weather


def attach_weather_features(meter_df: pd.DataFrame, weather_path: str | Path | None = None, seed: int = 42) -> pd.DataFrame:
    """Join weather context to meter readings by city and date.

    If no weather_path exists, a synthetic Albania weather dataset is generated
    over the date range of the meter data. If customer city is not available,
    Tirana is used as a safe fallback default.
    """
    data = meter_df.copy()
    if data.empty or "date" not in data.columns:
        return data

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if "city" not in data.columns:
        data["city"] = "Tirana"
    data["city"] = data["city"].fillna("Tirana").astype(str)

    meter_start = data["date"].min()
    meter_end = data["date"].max()
    meter_cities = sorted(data["city"].dropna().unique()) or ["Tirana"]

    if weather_path is not None and Path(weather_path).exists():
        weather = pd.read_csv(weather_path)
    else:
        weather = generate_albania_weather_dataset(str(meter_start.date()), str(meter_end.date()), cities=meter_cities, seed=seed)

    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    if "city" not in weather.columns:
        weather["city"] = "Tirana"
    weather["city"] = weather["city"].fillna("Tirana").astype(str)

    # If the bundled weather file does not cover a newly uploaded dataset
    # period/city, extend it deterministically instead of silently filling all
    # temperatures with neutral defaults.
    weather_min = weather["date"].min() if len(weather) else pd.NaT
    weather_max = weather["date"].max() if len(weather) else pd.NaT
    missing_city = not set(meter_cities).issubset(set(weather["city"].dropna().unique()))
    out_of_range = pd.isna(weather_min) or pd.isna(weather_max) or meter_start < weather_min or meter_end > weather_max
    if out_of_range or missing_city:
        generated = generate_albania_weather_dataset(str(meter_start.date()), str(meter_end.date()), cities=meter_cities, seed=seed)
        generated["date"] = pd.to_datetime(generated["date"], errors="coerce")
        weather = pd.concat([weather, generated], ignore_index=True)
        weather = weather.drop_duplicates(subset=["date", "city"], keep="first")

    # Avoid latitude/longitude collision with customer GIS coordinates.
    weather = weather.rename(columns={"latitude": "weather_latitude", "longitude": "weather_longitude"})
    keep_cols = [
        "date", "city", "temp_mean", "temp_min", "temp_max", "precipitation_mm",
        "heating_degree_days", "cooling_degree_days", "weather_class", "is_cold", "is_hot",
        "weather_demand_pressure", "weather_latitude", "weather_longitude",
    ]
    keep_cols = [c for c in keep_cols if c in weather.columns]
    out = data.merge(weather[keep_cols], on=["date", "city"], how="left")

    # Fallback: if a generated customer city is not in the weather file, use Tirana by date.
    missing_temp = out["temp_mean"].isna() if "temp_mean" in out.columns else pd.Series(True, index=out.index)
    if missing_temp.any():
        fallback = weather[weather["city"].eq("Tirana")][keep_cols].drop(columns=["city"], errors="ignore")
        fallback = fallback.add_prefix("fallback_")
        fallback = fallback.rename(columns={"fallback_date": "date"})
        out = out.merge(fallback, on="date", how="left")
        for col in [c for c in keep_cols if c not in ["date", "city"]]:
            fb = f"fallback_{col}"
            if col in out.columns and fb in out.columns:
                out[col] = out[col].fillna(out[fb])
        out = out.drop(columns=[c for c in out.columns if c.startswith("fallback_")], errors="ignore")

    # Robust default values if anything remains missing.
    for col, default in {
        "temp_mean": 18.0, "temp_min": 12.0, "temp_max": 24.0, "precipitation_mm": 0.0,
        "heating_degree_days": 0.0, "cooling_degree_days": 0.0, "weather_class": 0,
        "is_cold": 0, "is_hot": 0, "weather_demand_pressure": 0,
    }.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    return out
