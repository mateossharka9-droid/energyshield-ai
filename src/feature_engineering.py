"""Feature engineering focused on non-technical loss detection."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

CITY_CENTERS = {
    # Inland-safe operational centers. Coastal cities are shifted slightly inland
    # so prototype-generated meter points do not appear in the sea.
    "Tirana": (41.3275, 19.8187),
    "Durres": (41.3200, 19.4630),
    "Shkoder": (42.0693, 19.5200),
    "Vlore": (40.4705, 19.5080),
    "Elbasan": (41.1125, 20.0822),
    "Fier": (40.7239, 19.5680),
    "Korce": (40.6186, 20.7808),
    "Kukes": (42.0767, 20.4217),
}

# Operational zones tied to real Albanian cities. Zone names become the human-readable
# area_id values shown on every dashboard (e.g. "Tirana 2", "Durres 1") instead of opaque
# codes like "AREA_07". Larger cities get more zones; CITY_WEIGHT sets how many customers
# fall in each city so the distribution looks realistic.
CITY_ZONES = {
    "Tirana": 5, "Durres": 3, "Elbasan": 2, "Shkoder": 2,
    "Vlore": 2, "Fier": 1, "Korce": 1, "Kukes": 1,
}
CITY_WEIGHT = {
    "Tirana": 0.34, "Durres": 0.16, "Elbasan": 0.10, "Shkoder": 0.09,
    "Vlore": 0.09, "Fier": 0.08, "Korce": 0.08, "Kukes": 0.06,
}


def _build_city_zones() -> list[tuple[str, str]]:
    """Return [(area_name, parent_city)] using readable city-based zone names."""
    zones: list[tuple[str, str]] = []
    for city, n_zones in CITY_ZONES.items():
        for z in range(1, n_zones + 1):
            area_name = city if n_zones == 1 else f"{city} {z}"
            zones.append((area_name, city))
    return zones


def add_missing_operational_metadata(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add prototype GIS/operational fields if the uploaded dataset lacks them.

    Production note: these fields should come from GIS, transformer topology,
    meter registry, customer category registry, and inspection history.
    """
    rng = np.random.default_rng(seed)
    data = df.copy()
    customers = data["customer_id"].drop_duplicates().sort_values().tolist()
    n = len(customers)
    meta = pd.DataFrame({"customer_id": customers})

    if "customer_type" not in data.columns:
        meta["customer_type"] = rng.choice(["residential", "small_business", "industrial"], p=[0.84, 0.13, 0.03], size=n)
    # City-based geography: assign each customer to a real-city zone, then derive area,
    # transformer pool and coordinates from that same zone so everything stays coherent.
    zone_specs = _build_city_zones()  # list of (area_name, parent_city)
    zone_weights = np.array([CITY_WEIGHT[c] / CITY_ZONES[c] for _, c in zone_specs], dtype=float)
    zone_weights = zone_weights / zone_weights.sum()
    need_area = "area_id" not in data.columns
    need_transformer = "transformer_id" not in data.columns
    need_latlon = "latitude" not in data.columns or "longitude" not in data.columns
    zone_idx = (
        rng.choice(len(zone_specs), size=n, p=zone_weights)
        if (need_area or need_transformer or need_latlon) else None
    )

    if need_area:
        meta["area_id"] = [zone_specs[int(i)][0] for i in zone_idx]
    if need_transformer:
        # Each zone owns a small pool of substations/transformers (more for larger zones)
        # so transformer-level clusters stay inside a single city.
        zone_transformers, _t = {}, 1
        for zi in range(len(zone_specs)):
            k = max(3, int(round(70 * zone_weights[zi])))
            zone_transformers[zi] = [f"TR_{_t + j:03d}" for j in range(k)]
            _t += k
        meta["transformer_id"] = [str(rng.choice(zone_transformers[int(zi)])) for zi in zone_idx]
    if "meter_age" not in data.columns:
        meta["meter_age"] = rng.integers(1, 20, size=n)
    if "contract_power_kw" not in data.columns:
        meta["contract_power_kw"] = np.round(rng.uniform(3, 35, size=n), 1)

    # Billing / payment behavior — a strong real-world NTL signal. Customers who chronically
    # pay late, run up arrears, or have been disconnected for non-payment are materially more
    # likely to also tamper with metering. These fields are generated only when the upload does
    # not already provide them, so real OSHEE billing/collection data is respected when present.
    payment_fields = [
        "payment_late_count_12m", "unpaid_bills", "avg_payment_delay_days", "arrears_amount_lek",
        "disconnections_12m", "months_since_last_payment", "payment_method", "account_status",
    ]
    if not any(f in data.columns for f in payment_fields):
        if "fraud_label" in data.columns:
            fl_map = data[["customer_id", "fraud_label"]].drop_duplicates("customer_id").set_index("customer_id")["fraud_label"]
            fl = pd.to_numeric(pd.Series(customers).map(fl_map), errors="coerce").fillna(0).clip(0, 1).to_numpy()
        else:
            fl = np.zeros(n)
        # Latent payment-irregularity propensity (0..1): most customers are reliable payers,
        # a minority are chronically irregular, and fraud customers skew more irregular.
        prop = np.clip(rng.beta(1.7, 6.5, size=n) + fl * rng.uniform(0.10, 0.40, size=n), 0, 1)
        cycles = 12
        late = rng.binomial(cycles, np.clip(0.04 + 0.55 * prop, 0.0, 0.96))
        unpaid = np.minimum(late, rng.binomial(cycles, np.clip(0.02 + 0.42 * prop, 0.0, 0.92)))
        delay = np.round(np.clip(rng.normal(5 + 42 * prop, 8), 0, 150), 1)
        arrears = np.round(np.where(unpaid > 0, np.clip(rng.normal(3000 + 95000 * prop, 6000), 0, None), 0.0), 0)
        disconnects = rng.binomial(3, np.clip(0.015 + 0.32 * prop, 0.0, 0.85))
        months_since = np.round(np.clip(rng.normal(0.5 + 5.5 * prop, 1.3), 0, 18), 1)
        method = np.where(
            prop > 0.55, rng.choice(["cash_office", "none"], size=n, p=[0.7, 0.3]),
            np.where(prop > 0.30, rng.choice(["cash_office", "online"], size=n, p=[0.55, 0.45]),
                     rng.choice(["auto_debit", "online"], size=n, p=[0.5, 0.5])),
        )
        status = np.where(disconnects >= 2, "suspended", np.where((unpaid >= 3) | (months_since >= 4), "under_review", "active"))
        meta["payment_late_count_12m"] = late
        meta["unpaid_bills"] = unpaid
        meta["avg_payment_delay_days"] = delay
        meta["arrears_amount_lek"] = arrears
        meta["disconnections_12m"] = disconnects
        meta["months_since_last_payment"] = months_since
        meta["payment_method"] = method
        meta["account_status"] = status

    if need_latlon:
        if zone_idx is None:
            zone_idx = rng.choice(len(zone_specs), size=n, p=zone_weights)
        # Stable per-zone offset from the city center: zones of the same city sit next to
        # each other but stay visually distinct on the map.
        zone_offsets = {zi: rng.normal(0, 0.008, size=2) for zi in range(len(zone_specs))}
        meta["city"] = [zone_specs[int(zi)][1] for zi in zone_idx]
        lats, lons = [], []
        for zi in zone_idx:
            zi = int(zi)
            c_lat, c_lon = CITY_CENTERS[zone_specs[zi][1]]
            off = zone_offsets[zi]
            lat = c_lat + off[0] + rng.normal(0, 0.0026)
            lon = c_lon + off[1] + rng.normal(0, 0.0026)
            # Final defensive clipping around the city center keeps generated points
            # within a compact, land-oriented operating zone for the prototype map.
            lats.append(float(np.clip(lat, c_lat - 0.02, c_lat + 0.02)))
            lons.append(float(np.clip(lon, c_lon - 0.02, c_lon + 0.02)))
        meta["latitude"] = lats
        meta["longitude"] = lons
        meta["coordinate_source"] = "generated_albania_city_cluster"

    add_cols = [c for c in meta.columns if c not in data.columns]
    if add_cols:
        data = data.merge(meta[["customer_id"] + add_cols], on="customer_id", how="left")
    return data


def _longest_streak(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    values = mask.fillna(False).astype(bool).to_numpy()
    best = cur = 0
    for v in values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def build_daily_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling baselines and pattern flags to daily readings."""
    data = df.copy().sort_values(["customer_id", "date"])
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["season"] = np.select(
        [data["month"].isin([12, 1, 2]), data["month"].isin([3, 4, 5]), data["month"].isin([6, 7, 8])],
        ["winter", "spring", "summer"], default="autumn"
    )

    g = data.groupby("customer_id", group_keys=False)
    data["rolling_7d_mean"] = g["consumption_kwh"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    data["rolling_30d_mean"] = g["consumption_kwh"].transform(lambda s: s.rolling(30, min_periods=7).mean())
    data["rolling_30d_std"] = g["consumption_kwh"].transform(lambda s: s.rolling(30, min_periods=7).std())
    data["previous_30d_mean"] = g["rolling_30d_mean"].shift(7)
    data["drop_vs_previous_30d_pct"] = (data["previous_30d_mean"] - data["consumption_kwh"]) / data["previous_30d_mean"].replace(0, np.nan)
    data["spike_vs_previous_30d_pct"] = (data["consumption_kwh"] - data["previous_30d_mean"]) / data["previous_30d_mean"].replace(0, np.nan)
    data["sudden_drop_flag"] = (data["drop_vs_previous_30d_pct"] > 0.60).fillna(False).astype(int)
    data["sudden_spike_flag"] = (data["spike_vs_previous_30d_pct"] > 1.50).fillna(False).astype(int)
    data["zero_flag"] = (data["consumption_kwh"] <= 0.05).astype(int)

    # Low consumption relative to the customer's own median.
    customer_median = g["consumption_kwh"].transform("median").replace(0, np.nan)
    data["low_relative_flag"] = (data["consumption_kwh"] < customer_median * 0.25).fillna(False).astype(int)
    data["flatline_flag"] = ((data["rolling_30d_std"].fillna(999) < 0.05) & (data["rolling_30d_mean"].fillna(0) > 0)).astype(int)

    # Weather-aware NTL context. Hot/cold days often explain higher usage, so the
    # suspicious signal is a mismatch: unexpectedly low usage during high demand
    # weather, or a spike during normal weather.
    if "temp_mean" in data.columns:
        data["heating_degree_days"] = pd.to_numeric(data.get("heating_degree_days", 0), errors="coerce").fillna(0)
        data["cooling_degree_days"] = pd.to_numeric(data.get("cooling_degree_days", 0), errors="coerce").fillna(0)
        data["weather_demand_pressure"] = pd.to_numeric(data.get("weather_demand_pressure", 0), errors="coerce").fillna(0)
        data["weather_class"] = pd.to_numeric(data.get("weather_class", 0), errors="coerce").fillna(0).astype(int)
        data["temperature_abs_deviation"] = (pd.to_numeric(data["temp_mean"], errors="coerce").fillna(18.0) - 18.0).abs()
        data["weather_expected_high_usage"] = (data["weather_demand_pressure"] == 1).astype(int)
        data["low_on_extreme_weather_flag"] = ((data["weather_demand_pressure"] == 1) & (data["consumption_kwh"] < customer_median * 0.35)).fillna(False).astype(int)
        data["high_on_normal_weather_flag"] = ((data["weather_demand_pressure"] == 0) & (data["consumption_kwh"] > customer_median * 2.25) & (data["previous_30d_mean"].fillna(0) > 0)).fillna(False).astype(int)
        data["weather_consumption_mismatch"] = data[["low_on_extreme_weather_flag", "high_on_normal_weather_flag"]].max(axis=1)
    else:
        data["weather_demand_pressure"] = 0
        data["weather_class"] = 0
        data["temperature_abs_deviation"] = 0.0
        data["heating_degree_days"] = 0.0
        data["cooling_degree_days"] = 0.0
        data["is_cold"] = 0
        data["is_hot"] = 0
        data["low_on_extreme_weather_flag"] = 0
        data["high_on_normal_weather_flag"] = 0
        data["weather_consumption_mismatch"] = 0
    return data


def _trend_slope(s: pd.Series) -> float:
    y = s.to_numpy(dtype=float)
    if len(y) < 3 or np.allclose(y, y[0]):
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def build_customer_features(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily readings into customer-level NTL detection features."""
    data = daily.copy().sort_values(["customer_id", "date"])
    g = data.groupby("customer_id")

    agg = g.agg(
        total_consumption=("consumption_kwh", "sum"),
        avg_consumption=("consumption_kwh", "mean"),
        median_consumption=("consumption_kwh", "median"),
        std_consumption=("consumption_kwh", "std"),
        max_consumption=("consumption_kwh", "max"),
        min_consumption=("consumption_kwh", "min"),
        days_observed=("date", "nunique"),
        zero_days=("zero_flag", "sum"),
        low_days=("low_relative_flag", "sum"),
        sudden_drop_count=("sudden_drop_flag", "sum"),
        sudden_spike_count=("sudden_spike_flag", "sum"),
        flatline_days=("flatline_flag", "sum"),
        last_date=("date", "max"),
    ).reset_index()
    agg["std_consumption"] = agg["std_consumption"].fillna(0)
    agg["cv_consumption"] = agg["std_consumption"] / agg["avg_consumption"].replace(0, np.nan)
    agg["cv_consumption"] = agg["cv_consumption"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Recent vs previous period comparison.
    def _period_means(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("date")
        last30 = group.tail(30)["consumption_kwh"].mean()
        prev90 = group.iloc[max(0, len(group)-120):max(0, len(group)-30)]["consumption_kwh"].mean()
        first90 = group.head(min(90, len(group)))["consumption_kwh"].mean()
        return pd.Series({"last_30_mean": last30, "previous_90_mean": prev90, "first_90_mean": first90})

    periods = g.apply(_period_means, include_groups=False).reset_index()
    agg = agg.merge(periods, on="customer_id", how="left")
    base = agg["previous_90_mean"].fillna(agg["first_90_mean"]).replace(0, np.nan)
    agg["recent_drop_pct"] = ((base - agg["last_30_mean"]) / base).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    agg["recent_spike_pct"] = ((agg["last_30_mean"] - base) / base).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    agg["zero_day_ratio"] = agg["zero_days"] / agg["days_observed"].replace(0, np.nan)
    agg["low_day_ratio"] = agg["low_days"] / agg["days_observed"].replace(0, np.nan)
    agg["flatline_ratio"] = agg["flatline_days"] / agg["days_observed"].replace(0, np.nan)
    agg[["zero_day_ratio", "low_day_ratio", "flatline_ratio"]] = agg[["zero_day_ratio", "low_day_ratio", "flatline_ratio"]].fillna(0)

    weather_cols = {
        "avg_temp_mean": ("temp_mean", "mean"),
        "avg_heating_degree_days": ("heating_degree_days", "mean"),
        "avg_cooling_degree_days": ("cooling_degree_days", "mean"),
        "avg_weather_demand_pressure": ("weather_demand_pressure", "mean"),
        "weather_mismatch_days": ("weather_consumption_mismatch", "sum"),
        "low_extreme_weather_days": ("low_on_extreme_weather_flag", "sum"),
        "high_normal_weather_days": ("high_on_normal_weather_flag", "sum"),
        "cold_days": ("is_cold", "sum"),
        "hot_days": ("is_hot", "sum"),
    }
    available_weather_aggs = {k: v for k, v in weather_cols.items() if v[0] in data.columns}
    if available_weather_aggs:
        weather_agg = g.agg(**available_weather_aggs).reset_index()
        agg = agg.merge(weather_agg, on="customer_id", how="left")
    for c in ["weather_mismatch_days", "low_extreme_weather_days", "high_normal_weather_days", "cold_days", "hot_days"]:
        if c not in agg.columns:
            agg[c] = 0
    for c in ["avg_temp_mean", "avg_heating_degree_days", "avg_cooling_degree_days", "avg_weather_demand_pressure"]:
        if c not in agg.columns:
            agg[c] = 0.0
    agg["weather_mismatch_ratio"] = agg["weather_mismatch_days"] / agg["days_observed"].replace(0, np.nan)
    agg["extreme_weather_day_ratio"] = (agg["cold_days"] + agg["hot_days"]) / agg["days_observed"].replace(0, np.nan)
    agg[["weather_mismatch_ratio", "extreme_weather_day_ratio"]] = agg[["weather_mismatch_ratio", "extreme_weather_day_ratio"]].fillna(0)

    streaks = []
    slopes = []
    for customer_id, group in data.groupby("customer_id"):
        group = group.sort_values("date")
        streaks.append({
            "customer_id": customer_id,
            "longest_zero_streak": _longest_streak(group["zero_flag"] == 1),
            "longest_low_streak": _longest_streak(group["low_relative_flag"] == 1),
            "longest_weather_mismatch_streak": _longest_streak(group.get("weather_consumption_mismatch", pd.Series(0, index=group.index)) == 1),
        })
        slopes.append({"customer_id": customer_id, "consumption_trend_slope": _trend_slope(group["consumption_kwh"])})
    agg = agg.merge(pd.DataFrame(streaks), on="customer_id", how="left")
    agg = agg.merge(pd.DataFrame(slopes), on="customer_id", how="left")

    # Attach static metadata.
    meta_cols = [
        "customer_id", "customer_type", "area_id", "transformer_id", "latitude", "longitude",
        "city", "meter_age", "contract_power_kw", "fraud_label", "temp_mean", "weather_class",
        "payment_late_count_12m", "unpaid_bills", "avg_payment_delay_days", "arrears_amount_lek",
        "disconnections_12m", "months_since_last_payment", "payment_method", "account_status",
    ]
    available_meta = [c for c in meta_cols if c in data.columns]
    meta = data[available_meta].drop_duplicates("customer_id")
    agg = agg.merge(meta, on="customer_id", how="left")

    # Payment-behavior risk: billing discipline as an NTL trigger (late payments, arrears,
    # non-payment disconnections, dormant accounts). Defaults to 0 when no billing data exists,
    # so the score is fully backward-compatible with consumption-only uploads.
    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col in agg.columns:
            return pd.to_numeric(agg[col], errors="coerce").fillna(default)
        return pd.Series(default, index=agg.index, dtype=float)

    cycles = 12.0
    late = _num("payment_late_count_12m")
    unpaid = _num("unpaid_bills")
    delay = _num("avg_payment_delay_days")
    arrears = _num("arrears_amount_lek")
    disconnects = _num("disconnections_12m")
    months_since = _num("months_since_last_payment")
    if "account_status" in agg.columns:
        status = agg["account_status"].astype(str).str.lower()
    else:
        status = pd.Series("active", index=agg.index)
    agg["payment_on_time_ratio"] = ((cycles - late) / cycles).clip(0, 1)
    status_boost = np.where(status.str.contains("suspend"), 1.0, np.where(status.str.contains("review"), 0.5, 0.0))
    agg["payment_risk_score"] = np.clip((
        0.30 * np.minimum(late / cycles, 1) +
        0.22 * np.minimum(unpaid / 6.0, 1) +
        0.16 * np.minimum(delay / 60.0, 1) +
        0.12 * np.minimum(disconnects / 3.0, 1) +
        0.10 * np.minimum(arrears / 100000.0, 1) +
        0.10 * np.minimum(months_since / 6.0, 1)
    ) * 100 + 12 * status_boost, 0, 100)

    # Similar profile grouping. If no customer_type exists, all customers are same type.
    if "customer_type" not in agg.columns:
        agg["customer_type"] = "unknown"
    try:
        agg["consumption_band"] = pd.qcut(agg["first_90_mean"].rank(method="first"), q=5, labels=False) + 1
    except Exception:
        agg["consumption_band"] = 1
    peer_group = ["customer_type", "consumption_band"]
    peer_stats = agg.groupby(peer_group).agg(
        peer_avg_last_30=("last_30_mean", "mean"),
        peer_median_last_30=("last_30_mean", "median"),
        peer_avg_total=("total_consumption", "mean"),
        peer_count=("customer_id", "count"),
    ).reset_index()
    agg = agg.merge(peer_stats, on=peer_group, how="left")
    agg["peer_deviation_pct"] = ((agg["peer_avg_last_30"] - agg["last_30_mean"]) / agg["peer_avg_last_30"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    agg["customer_vs_peer_ratio"] = (agg["last_30_mean"] / agg["peer_avg_last_30"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1)

    # Pattern score is rule-based suspiciousness before ML.
    agg["sudden_behavior_score"] = (
        0.35 * np.minimum(agg["recent_drop_pct"], 1) +
        0.20 * np.minimum(agg["zero_day_ratio"] * 4, 1) +
        0.20 * np.minimum(agg["low_day_ratio"] * 3, 1) +
        0.15 * np.minimum(agg["flatline_ratio"] * 4, 1) +
        0.10 * np.minimum(agg["sudden_drop_count"] / 20, 1)
    ) * 100
    agg["weather_context_score"] = (
        0.55 * np.minimum(agg["weather_mismatch_ratio"] * 5, 1) +
        0.25 * np.minimum(agg["low_extreme_weather_days"] / 12, 1) +
        0.20 * np.minimum(agg["longest_weather_mismatch_streak"] / 10, 1)
    ) * 100

    # Fraud label is customer-level if available.
    if "fraud_label" in agg.columns:
        agg["fraud_label"] = pd.to_numeric(agg["fraud_label"], errors="coerce")

    return agg, data
