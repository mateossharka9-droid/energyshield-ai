"""Forecasting and operational audit helpers for EnergyShield NTL.

The forecasting layer creates a planning signal for inspection capacity. It is
not legal proof of fraud. The implementation deliberately uses robust, explainable
methods rather than a black-box sequence model because operators need stable,
interpretable next-period priorities from limited pilot data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


RISK_LEVEL_WEIGHT = {"Low": 0.2, "Medium": 0.55, "High": 0.85, "Critical": 1.0}


def _safe_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _robust_weekly_forecast(dates: pd.Series, values: pd.Series, steps: int = 30) -> np.ndarray:
    """Robust weekly-seasonal + recent-trend forecast.

    Why this method: electricity anomalies can be noisy, sparse and operationally
    expensive. A simple linear trend often creates unrealistic flat or explosive
    lines. This method blends same-weekday history, recent median behavior and a
    capped short trend, then clips to recent operational ranges.
    """
    y = _to_numeric_series(values).to_numpy(dtype=float)
    d = pd.to_datetime(dates, errors="coerce")
    valid = ~pd.isna(d)
    y = y[valid.to_numpy()]
    d = d[valid].reset_index(drop=True)
    if len(y) == 0:
        return np.zeros(steps)
    if len(y) < 10 or np.nanstd(y) < 1e-9:
        return np.repeat(float(np.nanmedian(y)), steps)

    hist = pd.DataFrame({"date": d, "value": y}).dropna().sort_values("date")
    hist["dow"] = hist["date"].dt.dayofweek
    recent = hist.tail(min(56, len(hist))).copy()
    recent14 = recent.tail(min(14, len(recent)))["value"]
    prev14 = recent.iloc[max(0, len(recent)-28):max(0, len(recent)-14)]["value"]
    recent_level = float(recent14.median()) if len(recent14) else float(hist["value"].median())
    prev_level = float(prev14.median()) if len(prev14) else recent_level
    daily_trend = np.clip((recent_level - prev_level) / 14.0, -0.08 * max(recent_level, 1.0), 0.08 * max(recent_level, 1.0))

    q05 = float(recent["value"].quantile(0.05))
    q95 = float(recent["value"].quantile(0.95))
    low_clip = max(0.0, q05 * 0.45)
    high_clip = max(q95 * 1.55, recent_level * 1.8, 1.0)

    last_date = hist["date"].max()
    preds = []
    for step in range(1, steps + 1):
        fdate = last_date + pd.Timedelta(days=step)
        same_dow = recent.loc[recent["dow"].eq(fdate.dayofweek), "value"]
        seasonal = float(same_dow.median()) if len(same_dow) else recent_level
        trended = recent_level + daily_trend * step
        pred = 0.56 * seasonal + 0.34 * recent_level + 0.10 * trended
        preds.append(float(np.clip(pred, low_clip, high_clip)))
    return np.asarray(preds, dtype=float)


def generate_forecasts(customer_scores: pd.DataFrame, area_scores: pd.DataFrame, daily_context: pd.DataFrame, output_dir: str | Path, horizon_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create loss, area, and customer risk forecasts for planning.

    Outputs:
    - loss_forecast.csv: daily planning forecast for next horizon_days
    - area_forecast.csv: next-period area ranking
    - customer_forecast.csv: customer-level next-period risk ranking
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    customer = customer_scores.copy()
    area = area_scores.copy()
    daily = daily_context.copy()

    current_month_loss = float(pd.to_numeric(customer.get("estimated_loss_all_30d", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    base_daily_loss = current_month_loss / max(horizon_days, 1)

    daily_loss_history = None
    hist_dates = None
    if not daily.empty and "date" in daily.columns:
        daily["date"] = _safe_date_series(daily["date"])
        for col in ["zero_flag", "sudden_drop_flag", "flatline_flag", "weather_consumption_mismatch", "consumption_kwh"]:
            if col not in daily.columns:
                daily[col] = 0
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0)
        daily_agg = daily.groupby("date", dropna=True).agg(
            average_consumption=("consumption_kwh", "mean"),
            zero_events=("zero_flag", "sum"),
            sudden_drop_events=("sudden_drop_flag", "sum"),
            flatline_events=("flatline_flag", "sum"),
            weather_mismatch_events=("weather_consumption_mismatch", "sum"),
        ).reset_index().dropna(subset=["date"]).sort_values("date")
        last_date = daily_agg["date"].max() if len(daily_agg) else pd.Timestamp.today().normalize()
        avg_pred = _robust_weekly_forecast(daily_agg["date"], daily_agg["average_consumption"], horizon_days)
        drop_pred = _robust_weekly_forecast(daily_agg["date"], daily_agg["sudden_drop_events"], horizon_days)
        zero_pred = _robust_weekly_forecast(daily_agg["date"], daily_agg["zero_events"], horizon_days)
        flat_pred = _robust_weekly_forecast(daily_agg["date"], daily_agg["flatline_events"], horizon_days)
        mismatch_pred = _robust_weekly_forecast(daily_agg["date"], daily_agg["weather_mismatch_events"], horizon_days)
        # Translate daily suspicious-event pressure into money using the current monetary loss as
        # the economic anchor, then forecast the money series directly so the prediction follows the
        # real recent trajectory (rising/falling), not a flat average.
        event_pressure = (
            0.95 * daily_agg["zero_events"] + 0.80 * daily_agg["sudden_drop_events"] +
            0.35 * daily_agg["flatline_events"] + 0.25 * daily_agg["weather_mismatch_events"]
        )
        recent_pressure = event_pressure.tail(30)
        recent_pressure_sum = float(event_pressure.tail(min(30, len(event_pressure))).sum())
        if recent_pressure_sum > 1e-9 and current_month_loss > 0:
            lek_per_pressure = current_month_loss / recent_pressure_sum
            daily_loss_history = (event_pressure * lek_per_pressure).to_numpy(dtype=float)
            hist_dates = daily_agg["date"]
    else:
        last_date = pd.Timestamp.today().normalize()
        avg_pred = np.repeat(float(pd.to_numeric(customer.get("avg_consumption", pd.Series([0])), errors="coerce").fillna(0).mean()), horizon_days)
        drop_pred = np.repeat(float(pd.to_numeric(customer.get("sudden_drop_count", pd.Series([0])), errors="coerce").fillna(0).sum() / max(horizon_days, 1)), horizon_days)
        zero_pred = np.repeat(float(pd.to_numeric(customer.get("zero_days", pd.Series([0])), errors="coerce").fillna(0).sum() / max(horizon_days, 1)), horizon_days)
        flat_pred = np.repeat(float(pd.to_numeric(customer.get("flatline_days", pd.Series([0])), errors="coerce").fillna(0).sum() / max(horizon_days, 1)), horizon_days)
        mismatch_pred = np.zeros(horizon_days)
        recent_pressure = pd.Series([float(drop_pred.mean() + zero_pred.mean() + flat_pred.mean())])

    if daily_loss_history is not None and len(daily_loss_history) >= 10 and float(np.nansum(daily_loss_history)) > 0:
        # Data-driven money forecast (captures trend + weekly seasonality), lightly anchored to the
        # current monthly loss for stability.
        money_pred = _robust_weekly_forecast(hist_dates, pd.Series(daily_loss_history), horizon_days)
        predicted_loss = 0.78 * money_pred + 0.22 * base_daily_loss
        cap = max(float(np.nanpercentile(daily_loss_history, 98)), base_daily_loss * 2.5, 1.0)
    else:
        predicted_event_pressure = 0.95 * zero_pred + 0.80 * drop_pred + 0.35 * flat_pred + 0.25 * mismatch_pred
        reference_pressure = max(float(np.nanmedian(recent_pressure)), 1.0)
        pressure_factor = np.clip(predicted_event_pressure / reference_pressure, 0.35, 2.25)
        weekday_shape = 1.0 + 0.05 * np.sin(2 * np.pi * np.arange(horizon_days) / 7.0)
        predicted_loss = base_daily_loss * (0.72 + 0.28 * pressure_factor) * weekday_shape
        cap = max(base_daily_loss * 2.0, float(np.nanmax(predicted_loss)) if len(predicted_loss) else 0, 1.0)
    predicted_loss = np.clip(predicted_loss, 0, cap)

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    total_events = drop_pred + zero_pred + flat_pred + mismatch_pred
    cumulative_loss = np.cumsum(predicted_loss)
    loss_forecast = pd.DataFrame({
        "date": future_dates,
        "predicted_average_consumption": np.round(avg_pred, 3),
        "predicted_sudden_drop_events": np.round(drop_pred, 2),
        "predicted_zero_consumption_events": np.round(zero_pred, 2),
        "predicted_flatline_events": np.round(flat_pred, 2),
        "predicted_weather_mismatch_events": np.round(mismatch_pred, 2),
        "predicted_total_events": np.round(total_events, 2),
        "predicted_loss_all": np.round(predicted_loss, 2),
        "cumulative_predicted_loss": np.round(cumulative_loss, 2),
        "forecast_lower_loss": np.round(predicted_loss * 0.82, 2),
        "forecast_upper_loss": np.round(predicted_loss * 1.18, 2),
        "planning_note": ["Capacity-planning signal; requires field verification before fraud decision."] * horizon_days,
    })
    loss_forecast.to_csv(output_dir / "loss_forecast.csv", index=False)

    # Area forecast: prioritize areas where current risk, anomaly density, loss and active weather mismatch converge.
    if not area.empty:
        af = area.copy()
        for col in ["area_risk_score", "anomaly_density", "estimated_loss_all_30d", "high_risk_customers", "critical_customers"]:
            if col not in af.columns:
                af[col] = 0
            af[col] = pd.to_numeric(af[col], errors="coerce").fillna(0)
        af["forecasted_area_risk_next_30d"] = (
            0.54 * af["area_risk_score"] +
            24 * af["anomaly_density"].clip(0, 1) +
            0.12 * (af["high_risk_customers"] / max(float(af["high_risk_customers"].max()), 1.0) * 100) +
            0.07 * (af["critical_customers"] / max(float(af["critical_customers"].max()), 1.0) * 100) +
            0.03 * (af["estimated_loss_all_30d"] / max(float(af["estimated_loss_all_30d"].max()), 1.0) * 100)
        ).clip(0, 100).round(2)
        af["forecast_priority"] = af["forecasted_area_risk_next_30d"].rank(method="first", ascending=False).astype(int)
        area_forecast = af.sort_values("forecast_priority")
    else:
        area_forecast = pd.DataFrame(columns=["area_id", "forecasted_area_risk_next_30d", "forecast_priority"])
    area_forecast.to_csv(output_dir / "area_forecast.csv", index=False)

    # Customer next-risk forecast: persistent risk + active recent signals + economic impact.
    cf = customer.copy()
    for col in ["risk_score", "recent_drop_pct", "zero_day_ratio", "flatline_ratio", "sudden_drop_count", "estimated_loss_all_30d", "weather_mismatch_ratio", "payment_risk_score"]:
        if col not in cf.columns:
            cf[col] = 0
        cf[col] = pd.to_numeric(cf[col], errors="coerce").fillna(0)
    trend_pressure = (
        0.24 * cf["recent_drop_pct"].clip(0, 1) * 100 +
        0.20 * cf["zero_day_ratio"].clip(0, 1) * 100 +
        0.16 * cf["flatline_ratio"].clip(0, 1) * 100 +
        0.16 * (cf["sudden_drop_count"] / max(float(cf["sudden_drop_count"].max()), 1.0) * 100) +
        0.12 * cf["weather_mismatch_ratio"].clip(0, 1) * 100 +
        0.12 * cf["payment_risk_score"].clip(0, 100)
    )
    economic_pressure = cf["estimated_loss_all_30d"] / max(float(cf["estimated_loss_all_30d"].max()), 1.0) * 100
    cf["forecasted_customer_risk_next_30d"] = (0.68 * cf["risk_score"] + 0.24 * trend_pressure + 0.08 * economic_pressure).clip(0, 100).round(2)
    cf["forecast_priority"] = cf["forecasted_customer_risk_next_30d"].rank(method="first", ascending=False).astype(int)
    customer_forecast = cf.sort_values("forecast_priority")[[c for c in [
        "forecast_priority", "customer_id", "risk_score", "forecasted_customer_risk_next_30d", "risk_level", "area_id", "transformer_id", "estimated_loss_all_30d", "main_reason", "recommended_action"
    ] if c in cf.columns]]
    customer_forecast.to_csv(output_dir / "customer_forecast.csv", index=False)

    return loss_forecast, area_forecast, customer_forecast


# The audit function is appended below from the previous implementation.

def build_operational_audit(customer: pd.DataFrame, area: pd.DataFrame, priority: pd.DataFrame, daily: pd.DataFrame, ingestion: dict | None = None) -> pd.DataFrame:
    """Find practical workflow gaps and missing features for OSHEE-style operations."""
    ingestion = ingestion or {}
    rows = []

    def add(area_name: str, finding: str, impact: str, improvement: str, priority_level: str = "Medium"):
        rows.append({
            "Audit Area": area_name,
            "Finding": finding,
            "Operational Impact": impact,
            "Recommended Improvement": improvement,
            "Priority": priority_level,
        })

    if customer.empty:
        add("Data coverage", "No customer risk output found.", "Analysts cannot prioritize inspections.", "Run Data Intake and NTL Detection first.", "Critical")
        return pd.DataFrame(rows)

    missing_geo = not {"latitude", "longitude"}.issubset(customer.columns) or customer[["latitude", "longitude"]].isna().any().any()
    if missing_geo:
        add("GIS integration", "Some customer coordinates are missing or synthetic.", "Maps and field routing may be inaccurate.", "Connect meter registry, address database, transformer GIS and OpenStreetMap geocoding.", "High")

    label_available = "fraud_label" in customer.columns and pd.to_numeric(customer["fraud_label"], errors="coerce").notna().sum() > 20
    if not label_available:
        add("Feedback loop", "Confirmed theft / false alarm labels are limited or missing.", "The supervised fraud probability model cannot learn from field outcomes.", "Save inspection results from Case Management and feed them back into the model every month.", "High")

    high_rate = float(customer.get("risk_level", pd.Series(dtype=str)).isin(["High", "Critical"]).mean() * 100)
    if high_rate > 25:
        add("Inspection capacity", f"High/Critical queue is large ({high_rate:.1f}% of customers).", "Field teams may not handle all alerts.", "Use forecast priority, area clustering, estimated loss and route planning to narrow the daily queue.", "High")
    elif high_rate < 1:
        add("Model sensitivity", f"High/Critical queue is very small ({high_rate:.1f}% of customers).", "The model may be too strict for exploratory detection.", "Review score thresholds and compare with known inspection outcomes.", "Medium")

    if not daily.empty and "date" in daily.columns:
        days = pd.to_datetime(daily["date"], errors="coerce").nunique()
        if days < 60:
            add("Historical depth", f"Only {days} days are available in the dashboard sample.", "Seasonal and trend detection may be weak.", "Use at least 6-12 months of meter readings for production NTL detection.", "High")
    else:
        add("Historical depth", "Daily consumption history is missing from dashboard outputs.", "Analysts cannot review customer timelines.", "Export daily history for high-risk and sampled normal customers.", "High")

    profile = ingestion.get("raw_profile", {}) if isinstance(ingestion, dict) else {}
    missing_pct = float(profile.get("missing_percent", 0) or 0)
    if missing_pct > 2:
        add("Data quality", f"Missing-cell rate is {missing_pct:.2f}%.", "Missing readings can create false anomalies.", "Add missing-reading reason codes, meter communication status and billing correction flags.", "Medium")

    if "estimated_loss_all_30d" in customer.columns and pd.to_numeric(customer["estimated_loss_all_30d"], errors="coerce").fillna(0).sum() <= 0:
        add("Financial impact", "Estimated loss is missing or zero.", "Managers cannot compare inspection value by case.", "Add tariff class, contract power, customer category and historical billing amount.", "Medium")

    if len(rows) == 0:
        add("Operational readiness", "No major blocking gap detected in the current prototype outputs.", "The platform can support a pilot workflow.", "Next step: connect real GIS, inspection outcomes, and role-based access.", "Medium")
    return pd.DataFrame(rows)
