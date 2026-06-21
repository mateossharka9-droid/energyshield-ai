"""End-to-end EnergyShield NTL pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .data_ingestion import ingest_dataset
from .feature_engineering import add_missing_operational_metadata, build_customer_features, build_daily_context
from .weather_features import attach_weather_features
from .geospatial import export_geospatial_outputs
from .modeling import add_expected_consumption_model, fit_anomaly_model, save_metrics
from .forecasting import generate_forecasts
from .risk_scoring import score_all
from .explainability import add_explanations


def run_ntl_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    model_dir: str | Path = "models",
    max_customers: Optional[int] = None,
    max_days: Optional[int] = None,
    seed: int = 42,
    weather_path: str | Path | None = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    clean_df, ingestion_report = ingest_dataset(
        input_path=input_path,
        output_dir=output_dir,
        max_customers=max_customers,
        max_days=max_days,
    )
    enriched = add_missing_operational_metadata(clean_df, seed=seed)
    default_weather = Path(__file__).resolve().parents[1] / "data" / "raw" / "albania_weather_daily_2014_2016.csv"
    selected_weather_path = weather_path if weather_path is not None else (default_weather if default_weather.exists() else None)
    enriched = attach_weather_features(enriched, weather_path=selected_weather_path, seed=seed)
    daily = build_daily_context(enriched)
    customer_features, daily_context = build_customer_features(daily)

    customer_features, expected_metrics = add_expected_consumption_model(customer_features, model_dir=model_dir)
    customer_features, model_metrics = fit_anomaly_model(customer_features, model_dir=model_dir)

    customer_scores, area_scores, transformer_scores, priority = score_all(customer_features)
    customer_scores = add_explanations(customer_scores)
    # priority needs explanations after customer_scores has them
    from .risk_scoring import create_inspection_priority
    priority = create_inspection_priority(customer_scores)

    customer_scores = customer_scores.sort_values("risk_score", ascending=False)
    area_scores = area_scores.sort_values("area_risk_score", ascending=False)
    transformer_scores = transformer_scores.sort_values("transformer_risk_score", ascending=False)

    customer_scores.to_csv(output_dir / "customer_risk_scores.csv", index=False)
    area_scores.to_csv(output_dir / "area_risk_scores.csv", index=False)
    transformer_scores.to_csv(output_dir / "transformer_risk_scores.csv", index=False)
    priority.to_csv(output_dir / "inspection_priority.csv", index=False)

    # Daily dashboard subset: all high-risk plus random low-risk sample for charts.
    high_ids = customer_scores.loc[customer_scores["risk_score"] >= 61, "customer_id"].head(500).tolist()
    if len(high_ids) < 30:
        high_ids += customer_scores.head(30)["customer_id"].tolist()
    daily_out = daily_context[daily_context["customer_id"].isin(set(high_ids))].copy()
    daily_out.to_csv(output_dir / "daily_dashboard.csv", index=False)

    export_geospatial_outputs(customer_scores, area_scores, priority, output_dir)
    loss_forecast, area_forecast, customer_forecast = generate_forecasts(customer_scores, area_scores, daily_context, output_dir)

    summary = {
        "customers_analyzed": int(customer_scores["customer_id"].nunique()),
        "records_analyzed": int(len(clean_df)),
        "date_min": str(clean_df["date"].min().date()) if len(clean_df) else None,
        "date_max": str(clean_df["date"].max().date()) if len(clean_df) else None,
        "high_risk_customers": int((customer_scores["risk_level"].isin(["High", "Critical"])).sum()),
        "critical_customers": int((customer_scores["risk_level"] == "Critical").sum()),
        "high_risk_areas": int((area_scores["area_risk_level"].isin(["High", "Critical"])).sum()),
        "estimated_loss_all_30d": float(customer_scores["estimated_loss_all_30d"].sum()),
        "top_area": area_scores.iloc[0]["area_id"] if len(area_scores) else None,
        "forecast_30d_loss_all": float(loss_forecast["predicted_loss_all"].sum()) if len(loss_forecast) else 0.0,
        "top_forecast_area": area_forecast.iloc[0]["area_id"] if len(area_forecast) and "area_id" in area_forecast.columns else None,
        "weather_context_enabled": bool("temp_mean" in daily_context.columns),
        "avg_weather_temperature": float(daily_context["temp_mean"].mean()) if "temp_mean" in daily_context.columns else None,
        "weather_extreme_days": int((daily_context.get("weather_demand_pressure", pd.Series(dtype=float)) == 1).sum()) if "weather_demand_pressure" in daily_context.columns else 0,
    }
    metrics = {"ingestion": ingestion_report, "expected_consumption": expected_metrics, "models": model_metrics, "summary": summary}
    save_metrics(metrics, output_dir)
    with open(output_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
