"""Synthetic SGCC-style data generator for controlled testing.

The challenge allows synthetic data for prototype testing. This generator creates
customer daily readings with injected NTL patterns: sudden drops, zeros, flatlines,
peer inconsistency, and geographic clusters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

CITY_CENTERS = {
    "Tirana": (41.3275, 19.8187),
    "Durres": (41.3200, 19.4630),
    "Shkoder": (42.0693, 19.5200),
    "Vlore": (40.4705, 19.5080),
    "Elbasan": (41.1125, 20.0822),
    "Fier": (40.7239, 19.5680),
    "Korce": (40.6186, 20.7808),
    "Kukes": (42.0767, 20.4217),
}

# Readable city-based zones (kept identical to src/feature_engineering.py) so synthetic
# test data and live ingested data share the same recognizable area names on the dashboards.
CITY_ZONES = {
    "Tirana": 5, "Durres": 3, "Elbasan": 2, "Shkoder": 2,
    "Vlore": 2, "Fier": 1, "Korce": 1, "Kukes": 1,
}
CITY_WEIGHT = {
    "Tirana": 0.34, "Durres": 0.16, "Elbasan": 0.10, "Shkoder": 0.09,
    "Vlore": 0.09, "Fier": 0.08, "Korce": 0.08, "Kukes": 0.06,
}


def generate_long_smart_meter_data(
    n_customers: int = 900,
    n_days: int = 365,
    fraud_rate: float = 0.09,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    customers = [f"C{100000+i}" for i in range(n_customers)]
    customer_type = rng.choice(["residential", "small_business", "industrial"], p=[0.82, 0.15, 0.03], size=n_customers)
    base_by_type = {"residential": 9.5, "small_business": 28.0, "industrial": 95.0}
    variability_by_type = {"residential": 2.2, "small_business": 6.5, "industrial": 18.0}

    zone_specs = []  # (area_name, parent_city)
    for city, n_zones in CITY_ZONES.items():
        for z in range(1, n_zones + 1):
            zone_specs.append((city if n_zones == 1 else f"{city} {z}", city))
    area_names = [a for a, _ in zone_specs]
    area_to_city = {a: c for a, c in zone_specs}
    zone_weights = np.array([CITY_WEIGHT[c] / CITY_ZONES[c] for _, c in zone_specs], dtype=float)
    zone_weights = zone_weights / zone_weights.sum()

    transformer_names = [f"TR_{i:03d}" for i in range(1, 71)]
    areas = rng.choice(area_names, size=n_customers, p=zone_weights)
    transformers = rng.choice(transformer_names, size=n_customers)

    cities = np.array([area_to_city[a] for a in areas])
    lat, lon = [], []
    for city in cities:
        c_lat, c_lon = CITY_CENTERS[city]
        lat.append(float(np.clip(c_lat + rng.normal(0, 0.008), c_lat - 0.02, c_lat + 0.02)))
        lon.append(float(np.clip(c_lon + rng.normal(0, 0.008), c_lon - 0.02, c_lon + 0.02)))

    fraud_label = rng.binomial(1, fraud_rate, size=n_customers)
    # Create a few geographic NTL clusters.
    cluster_areas = rng.choice(area_names, size=3, replace=False)
    cluster_mask = np.isin(areas, cluster_areas) & (rng.random(n_customers) < 0.25)
    fraud_label = np.where(cluster_mask, 1, fraud_label)

    meta = pd.DataFrame({
        "customer_id": customers,
        "customer_type": customer_type,
        "area_id": areas,
        "transformer_id": transformers,
        "city": cities,
        "latitude": lat,
        "longitude": lon,
        "meter_age": rng.integers(1, 18, size=n_customers),
        "contract_power_kw": np.round(rng.uniform(3, 35, size=n_customers), 1),
        "fraud_label": fraud_label,
    })

    records = []
    day_index = np.arange(n_days)
    # Albania-ish seasonal pattern: higher in winter and summer.
    yearly = 1.0 + 0.18 * np.cos(2 * np.pi * (day_index - 20) / 365) + 0.12 * np.cos(4 * np.pi * (day_index - 210) / 365)
    weekly = 1.0 + 0.06 * np.sin(2 * np.pi * day_index / 7)

    for i, cust in enumerate(customers):
        ctype = customer_type[i]
        base = max(1.0, rng.normal(base_by_type[ctype], variability_by_type[ctype]))
        noise = rng.normal(0, 0.12, size=n_days)
        trend = 1 + rng.normal(0, 0.0006) * day_index
        consumption = base * yearly * weekly * trend * (1 + noise)
        consumption = np.clip(consumption, 0, None)

        if fraud_label[i] == 1:
            pattern = rng.choice(["sudden_drop", "zero_period", "flatline", "peer_low"], p=[0.50, 0.18, 0.17, 0.15])
            start = int(rng.integers(low=max(30, n_days // 4), high=max(31, n_days - 45)))
            duration = int(rng.integers(28, min(120, n_days - start)))
            end = min(n_days, start + duration)

            if pattern == "sudden_drop":
                factor = rng.uniform(0.08, 0.38)
                consumption[start:end] *= factor
            elif pattern == "zero_period":
                consumption[start:end] = rng.uniform(0, 0.25, size=end - start)
            elif pattern == "flatline":
                flat_value = max(0.2, np.median(consumption[max(0, start-30):start]) * rng.uniform(0.10, 0.35))
                consumption[start:end] = flat_value + rng.normal(0, 0.03, size=end - start)
            elif pattern == "peer_low":
                consumption[int(n_days * 0.45):] *= rng.uniform(0.25, 0.55)

        for d, val in zip(dates, consumption):
            records.append((cust, d, round(float(val), 3), int(fraud_label[i])))

    df = pd.DataFrame(records, columns=["customer_id", "date", "consumption_kwh", "fraud_label"])
    df = df.merge(meta.drop(columns=["fraud_label"]), on="customer_id", how="left")
    return df, meta


def save_sgcc_wide_sample(path: str | Path, n_customers: int = 240, n_days: int = 180, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    long_df, meta = generate_long_smart_meter_data(n_customers=n_customers, n_days=n_days, seed=seed)
    wide = long_df.pivot_table(index="customer_id", columns="date", values="consumption_kwh", aggfunc="first")

    # Use portable date formatting.
    # NOTE: strftime("%-m") works on Linux/macOS but fails on Windows,
    # so we build SGCC-style dates manually as YYYY/M/D.
    formatted_columns = []
    for col in wide.columns:
        dt = pd.to_datetime(col)
        formatted_columns.append(f"{dt.year}/{dt.month}/{dt.day}")
    wide.columns = formatted_columns

    wide = wide.reset_index().rename(columns={"customer_id": "CONS_NO"})
    labels = meta[["customer_id", "fraud_label"]].rename(columns={"customer_id": "CONS_NO", "fraud_label": "FLAG"})
    wide = wide.merge(labels, on="CONS_NO", how="left")
    wide.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    save_sgcc_wide_sample("data/raw/sample_sgcc_wide_upload.csv")
