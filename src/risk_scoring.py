"""Risk scoring for customers and areas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip100(x):
    return np.clip(x, 0, 100)


# ERE-approved electricity tariff for Albania (residential, pre-VAT), in effect 2025-2026.
# Source: Enti Rregullator i Energjise (ERE) decision, effective 1 Feb 2025, held for 2026.
# Customers consuming up to 700 kWh/month pay the lower band; usage above that band
# is billed at the higher rate. Non-residential/business tariffs differ by contract and
# are approximated here with the higher band until real OSHEE billing tariffs are wired in.
TARIFF_LEK_PER_KWH_LOW_BAND = 8.5   # up to 700 kWh/month (covers ~95% of residential customers)
TARIFF_LEK_PER_KWH_HIGH_BAND = 9.5  # above 700 kWh/month, and default for non-residential
RESIDENTIAL_BAND_THRESHOLD_KWH_30D = 700.0


def _effective_tariff_lek_per_kwh(data: pd.DataFrame) -> pd.Series:
    """Pick the ERE tariff band per customer based on monthly consumption level.

    This is still a simplification (no progressive/partial-band billing, no VAT,
    no business-contract tariffs) but it reflects the real published OSHEE rates
    instead of an arbitrary flat EUR assumption, which is what an operator would
    immediately flag as unrealistic.
    """
    monthly_volume = pd.to_numeric(data.get("last_30_mean", 0), errors="coerce").fillna(0) * 30
    customer_type = data.get("customer_type", pd.Series(["residential"] * len(data), index=data.index)).astype(str).str.lower()
    is_residential = customer_type.str.contains("resid").fillna(True)
    band_rate = np.where(monthly_volume <= RESIDENTIAL_BAND_THRESHOLD_KWH_30D, TARIFF_LEK_PER_KWH_LOW_BAND, TARIFF_LEK_PER_KWH_HIGH_BAND)
    return pd.Series(np.where(is_residential, band_rate, TARIFF_LEK_PER_KWH_HIGH_BAND), index=data.index)


def _normalise_to_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    if s.max() - s.min() < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * 100


def compute_preliminary_customer_risk(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["historical_deviation_score"] = _clip100(
        55 * np.minimum(data.get("recent_drop_pct", 0), 1) +
        20 * np.minimum(data.get("low_day_ratio", 0) * 3, 1) +
        15 * np.minimum(data.get("zero_day_ratio", 0) * 4, 1) +
        10 * np.minimum(data.get("flatline_ratio", 0) * 4, 1)
    )
    data["peer_deviation_score"] = _clip100(100 * np.minimum(data.get("peer_deviation_pct", 0), 1))
    data["sudden_flags_score"] = _clip100(data.get("sudden_behavior_score", 0))
    data["context_deviation_score"] = _clip100(100 * np.minimum(data.get("expected_deviation_pct", 0), 1))
    data["weather_context_score"] = _clip100(data.get("weather_context_score", 0))
    data["payment_behavior_score"] = _clip100(data.get("payment_risk_score", 0))

    # If supervised probability exists, blend with anomaly score; otherwise only anomaly score.
    fraud_prob = pd.to_numeric(data.get("fraud_probability", np.nan), errors="coerce")
    ai_score = pd.to_numeric(data.get("ai_anomaly_score", 0), errors="coerce").fillna(0)
    data["ai_combined_score"] = np.where(fraud_prob.notna(), np.maximum(ai_score, fraud_prob.fillna(0)), ai_score)

    data["preliminary_risk_score"] = _clip100(
        0.32 * data["ai_combined_score"] +
        0.28 * data["historical_deviation_score"] +
        0.18 * data["peer_deviation_score"] +
        0.14 * data["sudden_flags_score"] +
        0.08 * data["weather_context_score"]
    )
    return data


def compute_area_and_transformer_risk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    if "area_id" not in data.columns:
        data["area_id"] = "UNKNOWN_AREA"
    if "transformer_id" not in data.columns:
        data["transformer_id"] = "UNKNOWN_TRANSFORMER"

    data["pre_high_risk_flag"] = (data["preliminary_risk_score"] >= 65).astype(int)
    data["pre_critical_flag"] = (data["preliminary_risk_score"] >= 82).astype(int)

    area = data.groupby("area_id").agg(
        customers=("customer_id", "count"),
        high_risk_customers=("pre_high_risk_flag", "sum"),
        critical_customers=("pre_critical_flag", "sum"),
        avg_preliminary_risk=("preliminary_risk_score", "mean"),
        avg_ai_anomaly=("ai_anomaly_score", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()
    area["anomaly_density"] = area["high_risk_customers"] / area["customers"].replace(0, np.nan)
    area_weighted = _clip100(
        0.50 * area["avg_preliminary_risk"] +
        0.40 * area["anomaly_density"].fillna(0) * 100 +
        0.10 * _normalise_to_100(area["critical_customers"])
    )
    # Geographic concentration should stand out even when individual scores are moderate.
    area["area_risk_score"] = _clip100(np.maximum(area_weighted, area["anomaly_density"].fillna(0) * 250))
    area["area_risk_level"] = pd.cut(
        area["area_risk_score"], bins=[-1, 30, 60, 80, 101], labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)

    trans = data.groupby("transformer_id").agg(
        customers=("customer_id", "count"),
        high_risk_customers=("pre_high_risk_flag", "sum"),
        avg_preliminary_risk=("preliminary_risk_score", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()
    trans["transformer_anomaly_density"] = trans["high_risk_customers"] / trans["customers"].replace(0, np.nan)
    trans_weighted = _clip100(
        0.60 * trans["avg_preliminary_risk"] + 0.40 * trans["transformer_anomaly_density"].fillna(0) * 100
    )
    trans["transformer_risk_score"] = _clip100(np.maximum(trans_weighted, trans["transformer_anomaly_density"].fillna(0) * 230))
    return area, trans


def compute_final_customer_risk(df: pd.DataFrame, area_scores: pd.DataFrame, transformer_scores: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.merge(area_scores[["area_id", "area_risk_score", "anomaly_density"]], on="area_id", how="left")
    data = data.merge(transformer_scores[["transformer_id", "transformer_risk_score", "transformer_anomaly_density"]], on="transformer_id", how="left")
    data[["area_risk_score", "transformer_risk_score", "anomaly_density", "transformer_anomaly_density"]] = data[["area_risk_score", "transformer_risk_score", "anomaly_density", "transformer_anomaly_density"]].fillna(0)

    data["geographic_risk_score"] = _clip100(0.65 * data["area_risk_score"] + 0.35 * data["transformer_risk_score"])

    # Exact weights aligned to challenge requirements. Billing/payment discipline is now an
    # explicit driver alongside consumption, peer, geographic and weather context signals.
    data["payment_behavior_score"] = _clip100(data.get("payment_behavior_score", data.get("payment_risk_score", 0)))
    base_risk = _clip100(
        0.26 * data["ai_combined_score"] +
        0.21 * data["historical_deviation_score"] +
        0.16 * data["peer_deviation_score"] +
        0.13 * data["geographic_risk_score"] +
        0.08 * data["sudden_flags_score"] +
        0.08 * data["weather_context_score"] +
        0.08 * data["payment_behavior_score"]
    )
    fraud_prob = pd.to_numeric(data.get("fraud_probability", np.nan), errors="coerce")
    supervised_risk = 0.65 * fraud_prob.fillna(0) + 0.35 * base_risk
    data["risk_score"] = _clip100(np.where(fraud_prob.notna(), np.maximum(base_risk, supervised_risk), base_risk))
    data["risk_level"] = pd.cut(
        data["risk_score"], bins=[-1, 30, 60, 80, 101], labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)

    # Estimated suspicious energy loss: expected recent - actual recent when actual is suspiciously lower.
    missing_kwh_day = (data.get("expected_last_30_consumption", data.get("previous_90_mean", 0)) - data.get("last_30_mean", 0)).clip(lower=0)
    data["estimated_missing_kwh_30d"] = missing_kwh_day * 30
    data["estimated_loss_all_30d"] = data["estimated_missing_kwh_30d"] * _effective_tariff_lek_per_kwh(data)  # ERE/OSHEE tariff bands, Lek pre-VAT
    return data


def create_inspection_priority(customer_scores: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    df = customer_scores.copy()
    df["inspection_priority_score"] = _clip100(
        0.55 * df["risk_score"] +
        0.25 * _normalise_to_100(df["estimated_loss_all_30d"]) +
        0.20 * df["geographic_risk_score"]
    )
    priority = df.sort_values("inspection_priority_score", ascending=False).head(top_n).copy()
    priority.insert(0, "priority_rank", range(1, len(priority) + 1))
    keep = [
        "priority_rank", "customer_id", "risk_score", "risk_level", "inspection_priority_score",
        "area_id", "transformer_id", "latitude", "longitude", "estimated_loss_all_30d",
        "weather_context_score", "weather_mismatch_ratio", "main_reason", "recommended_action",
    ]
    return priority[[c for c in keep if c in priority.columns]]


def score_all(customer_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prelim = compute_preliminary_customer_risk(customer_features)
    area, transformer = compute_area_and_transformer_risk(prelim)
    final = compute_final_customer_risk(prelim, area, transformer)
    # recompute area scores using final risk for exported view
    final["high_risk_flag"] = final["risk_score"].ge(61).astype(int)
    final["critical_flag"] = final["risk_score"].ge(81).astype(int)
    area_final = final.groupby("area_id").agg(
        customers=("customer_id", "count"),
        high_risk_customers=("high_risk_flag", "sum"),
        critical_customers=("critical_flag", "sum"),
        avg_risk_score=("risk_score", "mean"),
        avg_ai_anomaly=("ai_anomaly_score", "mean"),
        estimated_loss_all_30d=("estimated_loss_all_30d", "sum"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()
    area_final["anomaly_density"] = area_final["high_risk_customers"] / area_final["customers"].replace(0, np.nan)
    area_final_weighted = _clip100(
        0.50 * area_final["avg_risk_score"] + 0.35 * area_final["anomaly_density"].fillna(0) * 100 + 0.15 * _normalise_to_100(area_final["critical_customers"])
    )
    area_final["area_risk_score"] = _clip100(np.maximum(area_final_weighted, area_final["anomaly_density"].fillna(0) * 250))
    area_final["area_risk_level"] = pd.cut(
        area_final["area_risk_score"], bins=[-1, 30, 60, 80, 101], labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)
    priority = create_inspection_priority(final)
    return final, area_final, transformer, priority
