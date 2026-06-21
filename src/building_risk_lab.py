from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    IsolationForest = None
    RandomForestClassifier = None
    StandardScaler = None
    average_precision_score = None
    roc_auc_score = None
    train_test_split = None


ALBANIA_LOCATIONS: Dict[str, Tuple[float, float, str]] = {
    "Tirana": (41.3275, 19.8189, "City"),
    "Durres": (41.3231, 19.4414, "City"),
    "Shkoder": (42.0683, 19.5126, "City"),
    "Vlore": (40.4661, 19.4914, "City"),
    "Elbasan": (41.1125, 20.0822, "City"),
    "Fier": (40.7239, 19.5561, "City"),
    "Korce": (40.6186, 20.7808, "City"),
    "Berat": (40.7058, 19.9522, "City"),
    "Kukes": (42.0767, 20.4211, "City"),
    "Gjirokaster": (40.0758, 20.1389, "City"),
    "Peshkopi": (41.6850, 20.4289, "Village"),
    "Lushnje": (40.9419, 19.7050, "Village"),
    "Kruje": (41.5092, 19.7928, "Village"),
    "Permet": (40.2336, 20.3517, "Village"),
    "Pogradec": (40.9025, 20.6525, "Village"),
}

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


def _seasonal_temperature(dates: pd.Series, city: str, rng: np.random.Generator) -> pd.DataFrame:
    day = pd.to_datetime(dates).dt.dayofyear.to_numpy()
    # Albania-like climate: warmer coast, cooler mountain cities/villages.
    city_adj = {
        "Vlore": 2.2, "Durres": 1.6, "Fier": 1.1, "Tirana": .8, "Shkoder": .3,
        "Elbasan": .2, "Berat": .6, "Korce": -2.2, "Kukes": -2.0, "Peshkopi": -2.8,
        "Gjirokaster": .1, "Pogradec": -1.7, "Permet": -.2, "Kruje": .1, "Lushnje": .7,
    }.get(city, 0)
    base = 15.8 + city_adj + 11.8 * np.sin(2 * np.pi * (day - 172) / 365.25)
    temp_mean = base + rng.normal(0, 2.2, len(day))
    temp_min = temp_mean - rng.uniform(4.5, 8.5, len(day))
    temp_max = temp_mean + rng.uniform(5.0, 9.5, len(day))
    weather_class = np.where(temp_mean < 10, -1, np.where(temp_mean > 26, 1, 0))
    demand_pressure = np.abs(weather_class)
    heating_degree_days = np.maximum(0, 18 - temp_mean)
    cooling_degree_days = np.maximum(0, temp_mean - 24)
    return pd.DataFrame({
        "temp_mean": np.round(temp_mean, 2),
        "temp_min": np.round(temp_min, 2),
        "temp_max": np.round(temp_max, 2),
        "weather_class": weather_class.astype(int),
        "weather_demand_pressure": demand_pressure.astype(int),
        "heating_degree_days": np.round(heating_degree_days, 2),
        "cooling_degree_days": np.round(cooling_degree_days, 2),
    })


def generate_building_consumption_dataset(
    n_buildings: int = 80,
    days: int = 180,
    start_date: str = "2024-01-01",
    avg_units_per_building: int = 10,
    fraud_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic Albania building/apartment consumption dataset.

    The dataset is independent from the main NTL pipeline. It is meant for a
    separate module where operators can analyze apartments, floors, buildings,
    villages, and cities without changing the customer-level SGCC workflow.
    """
    rng = np.random.default_rng(seed)
    cities = list(ALBANIA_LOCATIONS.keys())
    date_index = pd.date_range(start=start_date, periods=int(days), freq="D")
    rows = []
    btypes = ["Residential", "Mixed Use", "Commercial", "Small Industrial"]
    meter_types = ["Smart", "Legacy", "Prepaid"]
    connection_types = ["Individual Meter", "Shared Meter", "Common Area", "Business Meter"]
    anomaly_types = ["none", "meter_bypass", "meter_tampering", "shared_illegal_connection", "under_reporting", "flatline_meter"]

    for b in range(1, int(n_buildings) + 1):
        city = rng.choice(cities, p=np.array([.19,.10,.08,.07,.08,.08,.06,.06,.05,.04,.05,.05,.04,.03,.02]))
        lat0, lon0, loc_type_default = ALBANIA_LOCATIONS[city]
        building_type = rng.choice(btypes, p=[.68,.18,.10,.04])
        location_type = loc_type_default if rng.random() > .20 else rng.choice(["City", "Village"])
        building_age = int(rng.integers(2, 55))
        floors = int(rng.integers(2, 13 if building_type != "Small Industrial" else 5))
        units = max(3, int(rng.normal(avg_units_per_building, max(2, avg_units_per_building * 0.35))))
        transformer_id = f"TR-{city[:3].upper()}-{rng.integers(100,999)}"
        building_id = f"BLD-{city[:3].upper()}-{b:04d}"
        # Small offsets around city centers. This avoids sea points while preserving a map feeling.
        lat = float(lat0 + rng.normal(0, 0.025 if location_type == "City" else 0.045))
        lon = float(lon0 + rng.normal(0, 0.025 if location_type == "City" else 0.045))
        lat = float(np.clip(lat, 39.65, 42.65))
        lon = float(np.clip(lon, 19.28, 21.05))

        # Building-level hidden issue: some buildings have illegal shared loss.
        building_has_shared_issue = rng.random() < (fraud_rate * 0.35)

        for u in range(1, units + 1):
            floor = int(rng.integers(0, max(1, floors) + 1))
            unit_id = f"{building_id}-U{u:03d}"
            unit_type = rng.choice(["Apartment", "Shop", "Office", "Workshop"], p=[.78,.10,.09,.03]) if building_type != "Small Industrial" else rng.choice(["Workshop","Office","Storage"], p=[.65,.2,.15])
            household_size = int(np.clip(rng.poisson(2.4) + 1, 1, 7)) if unit_type == "Apartment" else int(rng.integers(1, 5))
            area_sqm = float(np.clip(rng.normal(72, 24), 28, 180)) if unit_type == "Apartment" else float(np.clip(rng.normal(95, 45), 35, 320))
            meter_type = rng.choice(meter_types, p=[.58,.32,.10])
            connection_type = rng.choice(connection_types, p=[.82,.08,.04,.06])
            contracted_power_kw = float(np.clip(rng.normal(5.5, 1.6), 2.5, 18))
            base_kwh = (
                1.55 + household_size * 1.15 + area_sqm * 0.035 +
                (0.18 * floor) + (0.012 * building_age) +
                (1.5 if unit_type in ["Shop", "Office"] else 0) +
                (4.5 if unit_type == "Workshop" else 0)
            )
            # Legacy meters and shared meters have more irregular readings.
            noise_scale = 0.16 + (0.10 if meter_type == "Legacy" else 0) + (0.08 if connection_type == "Shared Meter" else 0)
            is_fraud = rng.random() < fraud_rate or (building_has_shared_issue and rng.random() < 0.35)
            anomaly_type = "none"
            if is_fraud:
                anomaly_type = rng.choice(anomaly_types[1:], p=[.24,.20,.18,.25,.13])

            weather = _seasonal_temperature(pd.Series(date_index), city, rng)
            for i, d in enumerate(date_index):
                season_pressure = 1 + 0.045 * weather.loc[i, "heating_degree_days"] + 0.075 * weather.loc[i, "cooling_degree_days"]
                weekend_factor = 1.08 if d.dayofweek >= 5 and unit_type == "Apartment" else (0.92 if d.dayofweek >= 5 else 1.0)
                expected = max(0.25, base_kwh * season_pressure * weekend_factor)
                actual = expected * rng.lognormal(mean=0, sigma=noise_scale)
                # Fraud patterns: usually under-reported, sometimes flatline, sometimes abrupt bypass.
                if anomaly_type == "meter_bypass":
                    if i > days * 0.35:
                        actual *= rng.uniform(.05, .28)
                elif anomaly_type == "meter_tampering":
                    if (i // 7) % 3 != 0:
                        actual *= rng.uniform(.18, .48)
                elif anomaly_type == "shared_illegal_connection":
                    actual *= rng.uniform(1.45, 2.35) if rng.random() < .24 else rng.uniform(.35, .70)
                elif anomaly_type == "under_reporting":
                    actual *= rng.uniform(.22, .58)
                elif anomaly_type == "flatline_meter":
                    if i > 8:
                        actual = base_kwh * rng.uniform(.08, .16)
                # Occasional reading noise.
                if rng.random() < .005:
                    actual = 0
                rows.append({
                    "date": d.date().isoformat(),
                    "city": city,
                    "location_type": location_type,
                    "building_id": building_id,
                    "building_type": building_type,
                    "building_age": building_age,
                    "floors": floors,
                    "floor": floor,
                    "unit_id": unit_id,
                    "unit_type": unit_type,
                    "household_size": household_size,
                    "area_sqm": round(area_sqm, 1),
                    "meter_type": meter_type,
                    "connection_type": connection_type,
                    "contracted_power_kw": round(contracted_power_kw, 1),
                    "transformer_id": transformer_id,
                    "latitude": round(lat + rng.normal(0, .0025), 6),
                    "longitude": round(lon + rng.normal(0, .0025), 6),
                    "expected_kwh": round(expected, 3),
                    "consumption_kwh": round(max(0, actual), 3),
                    "fraud_label": int(is_fraud),
                    "anomaly_type": anomaly_type,
                    **weather.loc[i].to_dict(),
                })
    return pd.DataFrame(rows)


def _read_any_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("ZIP file does not contain a CSV file.")
            with z.open(csvs[0]) as f:
                return pd.read_csv(f)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)


def _infer_column(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {str(c).lower().strip(): c for c in cols}
    for cand in candidates:
        c = cand.lower().strip()
        if c in lower:
            return lower[c]
    for cand in candidates:
        token = cand.lower().strip()
        for lc, orig in lower.items():
            if token in lc:
                return orig
    return None


def standardize_building_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize uploaded city/village/building consumption data.

    Required logical fields are date, building, unit/customer, and consumption.
    Missing metadata is inferred/filled so the module remains usable.
    """
    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty.")
    data = df.copy()
    cols = list(data.columns)
    mapping = {
        "date": _infer_column(cols, ["date", "reading_date", "invoice_date", "month", "time", "timestamp"]),
        "building_id": _infer_column(cols, ["building_id", "building", "block", "property_id", "site_id"]),
        "unit_id": _infer_column(cols, ["unit_id", "apartment_id", "flat_id", "customer_id", "client_id", "meter_id"]),
        "consumption_kwh": _infer_column(cols, ["consumption_kwh", "kwh", "energy_kwh", "consumption", "quantity", "reading"]),
        "floor": _infer_column(cols, ["floor", "storey", "level"]),
        "area_sqm": _infer_column(cols, ["area_sqm", "sqm", "surface", "m2", "apartment_area"]),
        "household_size": _infer_column(cols, ["household_size", "residents", "people", "occupants"]),
        "city": _infer_column(cols, ["city", "municipality", "area", "zone", "district"]),
        "location_type": _infer_column(cols, ["location_type", "urban_rural", "settlement_type"]),
        "latitude": _infer_column(cols, ["latitude", "lat"]),
        "longitude": _infer_column(cols, ["longitude", "lon", "lng"]),
        "fraud_label": _infer_column(cols, ["fraud_label", "is_fraud", "target", "label", "is_stealer", "theft"]),
        "expected_kwh": _infer_column(cols, ["expected_kwh", "expected_consumption", "baseline_kwh"]),
    }
    required = ["date", "building_id", "unit_id", "consumption_kwh"]
    missing = [k for k in required if mapping[k] is None]
    if missing:
        raise ValueError("Missing required logical columns: " + ", ".join(missing) + ". Include date, building_id, unit_id/customer_id, and consumption_kwh.")

    out = pd.DataFrame()
    for target, source in mapping.items():
        if source is not None:
            out[target] = data[source]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.date.astype(str)
    out["building_id"] = out["building_id"].astype(str)
    out["unit_id"] = out["unit_id"].astype(str)
    out["consumption_kwh"] = pd.to_numeric(out["consumption_kwh"], errors="coerce").clip(lower=0)
    out = out.dropna(subset=["consumption_kwh"])
    n = len(out)
    rng = np.random.default_rng(2026)
    if "city" not in out:
        out["city"] = rng.choice(list(ALBANIA_LOCATIONS.keys()), size=n)
    if "location_type" not in out:
        out["location_type"] = np.where(out["city"].map(lambda x: ALBANIA_LOCATIONS.get(str(x), (0,0,"City"))[2]) == "Village", "Village", "City")
    if "floor" not in out:
        out["floor"] = rng.integers(0, 8, size=n)
    if "area_sqm" not in out:
        out["area_sqm"] = np.round(rng.normal(75, 24, size=n).clip(25, 220), 1)
    if "household_size" not in out:
        out["household_size"] = rng.integers(1, 6, size=n)
    if "building_type" not in out:
        out["building_type"] = "Residential"
    if "unit_type" not in out:
        out["unit_type"] = "Apartment"
    if "meter_type" not in out:
        out["meter_type"] = "Unknown"
    if "connection_type" not in out:
        out["connection_type"] = "Individual Meter"
    if "building_age" not in out:
        out["building_age"] = rng.integers(3, 45, size=n)
    if "floors" not in out:
        out["floors"] = out.groupby("building_id")["floor"].transform("max").fillna(out["floor"]).astype(int).clip(lower=1)
    if "contracted_power_kw" not in out:
        out["contracted_power_kw"] = np.round(rng.normal(5.5, 1.7, size=n).clip(2, 16), 1)
    if "transformer_id" not in out:
        out["transformer_id"] = out["city"].astype(str).str[:3].str.upper().radd("TR-")
    if "latitude" not in out or "longitude" not in out:
        coords = out["city"].map(lambda c: ALBANIA_LOCATIONS.get(str(c), ALBANIA_LOCATIONS["Tirana"]))
        out["latitude"] = [float(x[0]) for x in coords] + rng.normal(0, 0.025, size=n)
        out["longitude"] = [float(x[1]) for x in coords] + rng.normal(0, 0.025, size=n)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce").fillna(41.3275).clip(39.65, 42.65)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce").fillna(19.8189).clip(19.28, 21.05)
    # Weather context if missing.
    if "temp_mean" not in out:
        chunks = []
        for city, part in out.groupby("city", sort=False):
            weather = _seasonal_temperature(pd.to_datetime(part["date"]), str(city), rng)
            chunks.append(pd.DataFrame(weather.values, index=part.index, columns=weather.columns))
        weather_all = pd.concat(chunks).sort_index()
        for c in weather_all.columns:
            out[c] = weather_all[c]
    if "expected_kwh" not in out:
        out["expected_kwh"] = (
            1.45 + pd.to_numeric(out["household_size"], errors="coerce").fillna(2.5) * 1.05 +
            pd.to_numeric(out["area_sqm"], errors="coerce").fillna(70) * 0.037 +
            pd.to_numeric(out["floor"], errors="coerce").fillna(2) * 0.12 +
            pd.to_numeric(out.get("heating_degree_days", 0), errors="coerce").fillna(0) * 0.24 +
            pd.to_numeric(out.get("cooling_degree_days", 0), errors="coerce").fillna(0) * 0.38
        )
    if "fraud_label" in out:
        out["fraud_label"] = pd.to_numeric(out["fraud_label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        out["fraud_label"] = 0
    if "anomaly_type" not in out:
        out["anomaly_type"] = np.where(out["fraud_label"] == 1, "uploaded_positive_label", "unknown")
    return out


def _safe_pct(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def _minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def _risk_level(score: float) -> str:
    if score >= 82:
        return "Critical"
    if score >= 64:
        return "High"
    if score >= 38:
        return "Medium"
    return "Low"


def analyze_building_dataset(df: pd.DataFrame, output_dir: str | Path | None = None, seed: int = 42) -> Dict[str, pd.DataFrame | dict | str]:
    data = standardize_building_dataset(df)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["unit_id", "date"])
    for c in ["consumption_kwh", "expected_kwh", "area_sqm", "household_size", "floor"]:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    data["consumption_gap_pct"] = (data["expected_kwh"] - data["consumption_kwh"]) / data["expected_kwh"].replace(0, np.nan)
    data["weather_low_mismatch"] = ((pd.to_numeric(data.get("weather_demand_pressure", 0), errors="coerce").fillna(0) >= 1) & (data["consumption_kwh"] < data["expected_kwh"] * 0.35)).astype(int)
    data["normal_high_mismatch"] = ((pd.to_numeric(data.get("weather_demand_pressure", 0), errors="coerce").fillna(0) == 0) & (data["consumption_kwh"] > data["expected_kwh"] * 1.85)).astype(int)
    data["zero_flag"] = (data["consumption_kwh"] <= 0.05).astype(int)
    data["low_expected_flag"] = (data["consumption_kwh"] < data["expected_kwh"] * 0.25).astype(int)
    data["prev_kwh"] = data.groupby("unit_id")["consumption_kwh"].shift(1)
    data["sudden_drop_flag"] = ((data["prev_kwh"] > 2) & (data["consumption_kwh"] < data["prev_kwh"] * 0.28)).astype(int)
    data["small_change"] = (data.groupby("unit_id")["consumption_kwh"].diff().abs() < 0.04).astype(int)

    agg = data.groupby("unit_id").agg(
        building_id=("building_id", "first"),
        city=("city", "first"),
        location_type=("location_type", "first"),
        building_type=("building_type", "first"),
        unit_type=("unit_type", "first"),
        meter_type=("meter_type", "first"),
        connection_type=("connection_type", "first"),
        transformer_id=("transformer_id", "first"),
        floor=("floor", "first"),
        floors=("floors", "first"),
        area_sqm=("area_sqm", "first"),
        household_size=("household_size", "first"),
        contracted_power_kw=("contracted_power_kw", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        readings=("consumption_kwh", "size"),
        avg_consumption=("consumption_kwh", "mean"),
        median_consumption=("consumption_kwh", "median"),
        std_consumption=("consumption_kwh", "std"),
        total_consumption=("consumption_kwh", "sum"),
        avg_expected_kwh=("expected_kwh", "mean"),
        total_expected_kwh=("expected_kwh", "sum"),
        zero_days=("zero_flag", "sum"),
        low_expected_days=("low_expected_flag", "sum"),
        sudden_drop_days=("sudden_drop_flag", "sum"),
        flatline_days=("small_change", "sum"),
        weather_low_mismatch_days=("weather_low_mismatch", "sum"),
        normal_high_mismatch_days=("normal_high_mismatch", "sum"),
        avg_temp_mean=("temp_mean", "mean"),
        avg_weather_demand_pressure=("weather_demand_pressure", "mean"),
        fraud_label=("fraud_label", "max"),
        anomaly_type=("anomaly_type", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
    ).reset_index()
    agg["std_consumption"] = agg["std_consumption"].fillna(0)
    agg["zero_ratio"] = agg["zero_days"] / agg["readings"].replace(0, np.nan)
    agg["flatline_ratio"] = agg["flatline_days"] / agg["readings"].replace(0, np.nan)
    agg["low_expected_ratio"] = agg["low_expected_days"] / agg["readings"].replace(0, np.nan)
    agg["weather_mismatch_ratio"] = (agg["weather_low_mismatch_days"] + agg["normal_high_mismatch_days"]) / agg["readings"].replace(0, np.nan)
    agg["expected_gap_pct"] = (agg["avg_expected_kwh"] - agg["avg_consumption"]) / agg["avg_expected_kwh"].replace(0, np.nan)
    agg["kwh_per_sqm"] = agg["avg_consumption"] / agg["area_sqm"].replace(0, np.nan)
    agg["kwh_per_person"] = agg["avg_consumption"] / agg["household_size"].replace(0, np.nan)
    agg["cv_consumption"] = agg["std_consumption"] / agg["avg_consumption"].replace(0, np.nan)

    peer_cols = ["city", "location_type", "unit_type"]
    peer = agg.groupby(peer_cols)["avg_consumption"].transform("median")
    fallback_peer = agg.groupby(["location_type", "unit_type"])["avg_consumption"].transform("median")
    agg["peer_avg_consumption"] = peer.fillna(fallback_peer).fillna(agg["avg_consumption"].median())
    agg["peer_low_gap_pct"] = (agg["peer_avg_consumption"] - agg["avg_consumption"]) / agg["peer_avg_consumption"].replace(0, np.nan)
    agg["peer_low_gap_pct"] = agg["peer_low_gap_pct"].clip(lower=0).fillna(0)
    agg["estimated_missing_kwh_30d"] = np.maximum(0, agg["avg_expected_kwh"] - agg["avg_consumption"]) * 30
    agg["estimated_loss_lek_30d"] = agg["estimated_missing_kwh_30d"] * 13.5

    feature_cols = [
        "avg_consumption", "std_consumption", "zero_ratio", "flatline_ratio", "low_expected_ratio",
        "sudden_drop_days", "weather_mismatch_ratio", "expected_gap_pct", "peer_low_gap_pct",
        "kwh_per_sqm", "kwh_per_person", "cv_consumption", "avg_weather_demand_pressure",
        "area_sqm", "household_size", "floor", "contracted_power_kw",
    ]
    X = agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    metrics = {"source": "Building/Village Risk Lab", "records": int(len(data)), "units": int(len(agg)), "buildings": int(agg["building_id"].nunique())}
    if IsolationForest is not None and len(X) >= 10:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(n_estimators=180, contamination="auto", random_state=seed)
        iso.fit(Xs)
        raw = -iso.score_samples(Xs)
        agg["building_ai_anomaly_score"] = (_minmax(pd.Series(raw, index=agg.index)) * 100).round(2)
    else:
        agg["building_ai_anomaly_score"] = 0

    # Supervised probability if uploaded/generated labels have both classes.
    agg["fraud_probability"] = np.nan
    label_counts = agg["fraud_label"].value_counts().to_dict()
    metrics["positive_units"] = int(label_counts.get(1, 0))
    if RandomForestClassifier is not None and agg["fraud_label"].nunique() == 2 and len(agg) >= 40 and min(label_counts.values()) >= 5:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, agg["fraud_label"], test_size=0.28, stratify=agg["fraud_label"], random_state=seed)
            clf = RandomForestClassifier(n_estimators=260, max_depth=8, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs_test = clf.predict_proba(X_test)[:, 1]
            metrics["fraud_classifier_roc_auc"] = float(roc_auc_score(y_test, probs_test)) if len(np.unique(y_test)) == 2 else None
            metrics["fraud_classifier_average_precision"] = float(average_precision_score(y_test, probs_test))
            agg["fraud_probability"] = clf.predict_proba(X)[:, 1]
            importances = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
        except Exception as exc:
            metrics["classifier_warning"] = str(exc)
            importances = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    else:
        importances = pd.DataFrame({"feature": feature_cols, "importance": np.nan})

    agg["consumption_deviation_score"] = (np.maximum(0, agg["expected_gap_pct"]).clip(0, 1.2) / 1.2 * 100).round(2)
    agg["peer_deviation_score"] = (agg["peer_low_gap_pct"].clip(0, 1) * 100).round(2)
    agg["meter_behavior_score"] = ((agg["zero_ratio"] * 42 + agg["flatline_ratio"] * 34 + np.minimum(agg["sudden_drop_days"], 12) / 12 * 24).clip(0, 100)).round(2)
    agg["weather_context_score"] = (agg["weather_mismatch_ratio"].clip(0, 1) * 100).round(2)
    agg["loss_impact_score"] = (_minmax(agg["estimated_loss_lek_30d"]) * 100).round(2)
    base_risk = (
        0.23 * agg["building_ai_anomaly_score"] +
        0.22 * agg["consumption_deviation_score"] +
        0.18 * agg["peer_deviation_score"] +
        0.17 * agg["meter_behavior_score"] +
        0.12 * agg["weather_context_score"] +
        0.08 * agg["loss_impact_score"]
    )
    agg["risk_score"] = base_risk.clip(0, 100).round(2)
    # If a supervised model exists, blend probability with transparent risk score, not replace it.
    if agg["fraud_probability"].notna().any():
        agg["risk_score"] = (0.68 * agg["risk_score"] + 0.32 * (agg["fraud_probability"] * 100)).clip(0, 100).round(2)
    else:
        agg["fraud_probability"] = (1 / (1 + np.exp(-(agg["risk_score"] - 55) / 12))).round(4)
    agg["risk_level"] = agg["risk_score"].apply(_risk_level)
    drivers = ["consumption_deviation_score", "peer_deviation_score", "meter_behavior_score", "weather_context_score", "building_ai_anomaly_score", "loss_impact_score"]
    driver_names = {
        "consumption_deviation_score":"below expected consumption",
        "peer_deviation_score":"below similar units",
        "meter_behavior_score":"meter flatline/drop/zero behavior",
        "weather_context_score":"weather-consumption mismatch",
        "building_ai_anomaly_score":"AI outlier behavior",
        "loss_impact_score":"estimated loss impact",
    }
    top_driver = agg[drivers].idxmax(axis=1)
    agg["main_reason"] = top_driver.map(driver_names)
    agg["recommended_action"] = np.where(
        agg["risk_level"].isin(["Critical", "High"]),
        "Prioritize meter inspection, compare common-area meter, and verify illegal shared connections.",
        "Monitor trend and inspect only if risk persists or neighboring units also become suspicious.",
    )
    agg["alert_explanation"] = agg.apply(
        lambda r: f"{r['unit_id']} in {r['building_id']} is {r['risk_level']} risk because the strongest signal is {r['main_reason']}. Expected gap: {r['expected_gap_pct']:.0%}, peer low gap: {r['peer_low_gap_pct']:.0%}, weather mismatch: {r['weather_mismatch_ratio']:.0%}.",
        axis=1,
    )
    unit_scores = agg.sort_values("risk_score", ascending=False).reset_index(drop=True)
    unit_scores.insert(0, "priority_rank", np.arange(1, len(unit_scores)+1))

    building_scores = unit_scores.groupby("building_id").agg(
        city=("city", "first"),
        location_type=("location_type", "first"),
        building_type=("building_type", "first"),
        transformer_id=("transformer_id", "first"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        units=("unit_id", "nunique"),
        high_risk_units=("risk_level", lambda x: int(pd.Series(x).isin(["High","Critical"]).sum())),
        critical_units=("risk_level", lambda x: int((pd.Series(x) == "Critical").sum())),
        avg_risk_score=("risk_score", "mean"),
        max_risk_score=("risk_score", "max"),
        p90_risk_score=("risk_score", lambda x: float(np.percentile(x, 90))),
        avg_fraud_probability=("fraud_probability", "mean"),
        estimated_loss_lek_30d=("estimated_loss_lek_30d", "sum"),
        top_reason=("main_reason", lambda x: x.mode().iat[0] if not x.mode().empty else "N/A"),
    ).reset_index()
    building_scores["risk_score"] = (0.55 * building_scores["p90_risk_score"] + 0.30 * building_scores["avg_risk_score"] + 0.15 * np.minimum(building_scores["high_risk_units"] / building_scores["units"].replace(0, np.nan) * 100, 100)).clip(0, 100).round(2)
    building_scores["risk_level"] = building_scores["risk_score"].apply(_risk_level)
    building_scores = building_scores.sort_values("risk_score", ascending=False).reset_index(drop=True)
    building_scores.insert(0, "priority_rank", np.arange(1, len(building_scores)+1))

    floor_scores = unit_scores.groupby(["building_id", "floor"]).agg(
        city=("city", "first"), units=("unit_id", "nunique"), avg_risk_score=("risk_score", "mean"),
        max_risk_score=("risk_score", "max"), high_risk_units=("risk_level", lambda x: int(pd.Series(x).isin(["High","Critical"]).sum())),
        estimated_loss_lek_30d=("estimated_loss_lek_30d", "sum"),
    ).reset_index().sort_values(["avg_risk_score", "high_risk_units"], ascending=False)

    daily_summary = data.groupby("date").agg(
        total_consumption_kwh=("consumption_kwh", "sum"),
        expected_consumption_kwh=("expected_kwh", "sum"),
        weather_mismatch_events=("weather_low_mismatch", "sum"),
        zero_events=("zero_flag", "sum"),
        sudden_drop_events=("sudden_drop_flag", "sum"),
        avg_temperature=("temp_mean", "mean"),
    ).reset_index()
    daily_summary["estimated_missing_kwh"] = np.maximum(0, daily_summary["expected_consumption_kwh"] - daily_summary["total_consumption_kwh"])

    metrics.update({
        "high_risk_units": int(unit_scores["risk_level"].isin(["High", "Critical"]).sum()),
        "critical_units": int((unit_scores["risk_level"] == "Critical").sum()),
        "high_risk_buildings": int(building_scores["risk_level"].isin(["High", "Critical"]).sum()),
        "estimated_loss_lek_30d": float(unit_scores["estimated_loss_lek_30d"].sum()),
        "date_min": str(data["date"].min().date()),
        "date_max": str(data["date"].max().date()),
        "scoring_method": "Hybrid: expected-consumption deviation + peer comparison + meter behavior + weather mismatch + IsolationForest, with supervised probability if labels exist.",
    })

    report = build_building_report(unit_scores, building_scores, floor_scores, metrics)
    results = {
        "raw": data,
        "unit_scores": unit_scores,
        "building_scores": building_scores,
        "floor_scores": floor_scores,
        "daily_summary": daily_summary,
        "feature_importance": importances,
        "metrics": metrics,
        "report": report,
    }
    if output_dir is not None:
        save_building_outputs(results, output_dir)
    return results


def build_building_report(unit_scores: pd.DataFrame, building_scores: pd.DataFrame, floor_scores: pd.DataFrame, metrics: dict) -> str:
    high_units = int(metrics.get("high_risk_units", 0))
    high_buildings = int(metrics.get("high_risk_buildings", 0))
    loss = float(metrics.get("estimated_loss_lek_30d", 0))
    top_b = building_scores.head(5)
    top_u = unit_scores.head(5)
    lines = [
        "Building / Village Risk Lab - Operational Report",
        "================================================",
        f"Analysis period: {metrics.get('date_min','N/A')} to {metrics.get('date_max','N/A')}",
        f"Buildings analyzed: {metrics.get('buildings', 0):,}",
        f"Units analyzed: {metrics.get('units', 0):,}",
        f"High/Critical units: {high_units:,}",
        f"High/Critical buildings: {high_buildings:,}",
        f"Estimated 30-day suspicious loss: {loss:,.0f} Lek",
        "",
        "Recommended operational use:",
        "- Use this module as a separate building/city/village investigation layer.",
        "- Prioritize buildings with several high-risk units before individual low-confidence cases.",
        "- Verify common-area meters, shared connections, meter bypass, and floor-level clusters.",
        "- Feed confirmed inspection outcomes back as labels for supervised calibration.",
        "",
        "Top buildings:",
    ]
    for _, r in top_b.iterrows():
        lines.append(f"- {r['building_id']} ({r['city']}): risk {r['risk_score']:.1f}, high-risk units {int(r['high_risk_units'])}/{int(r['units'])}, reason: {r['top_reason']}")
    lines.append("")
    lines.append("Top units:")
    for _, r in top_u.iterrows():
        lines.append(f"- {r['unit_id']}: risk {r['risk_score']:.1f}, probability {float(r['fraud_probability']):.1%}, reason: {r['main_reason']}")
    return "\n".join(lines)


def save_building_outputs(results: Dict[str, pd.DataFrame | dict | str], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["raw", "unit_scores", "building_scores", "floor_scores", "daily_summary", "feature_importance"]:
        obj = results.get(name)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(out / f"{name}.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results.get("metrics", {}), f, indent=2)
    with open(out / "operational_report.txt", "w", encoding="utf-8") as f:
        f.write(str(results.get("report", "")))


def load_building_outputs(output_dir: str | Path) -> Dict[str, pd.DataFrame | dict | str]:
    out = Path(output_dir)
    results: Dict[str, pd.DataFrame | dict | str] = {}
    for name in ["raw", "unit_scores", "building_scores", "floor_scores", "daily_summary", "feature_importance"]:
        p = out / f"{name}.csv"
        results[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    p = out / "metrics.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            results["metrics"] = json.load(f)
    else:
        results["metrics"] = {}
    rp = out / "operational_report.txt"
    results["report"] = rp.read_text(encoding="utf-8") if rp.exists() else ""
    return results


def read_and_standardize_building_file(path: str | Path) -> pd.DataFrame:
    return standardize_building_dataset(_read_any_table(path))
