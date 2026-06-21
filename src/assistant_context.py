"""Shared grounding logic for the Operations Assistant.

Defines the system instruction and the JSON context snapshot used to ground
the Gemini-backed chatbot in the platform's own computed results. This file
has zero dependency on any LLM SDK.
"""

from __future__ import annotations

import json

import pandas as pd

SYSTEM_INSTRUCTION = (
    "You are the Operations Assistant inside EnergyShield AI, a non-technical electricity "
    "loss (NTL) investigation platform used by OSHEE (Albania's electricity distribution "
    "operator). You help administrators, analysts, and field inspectors interpret the "
    "platform's own computed results.\n\n"
    "Rules you must always follow:\n"
    "1. Only use facts present in the JSON context provided with each question. Never invent "
    "customer IDs, risk scores, area names, or loss figures that are not in the context.\n"
    "2. If the answer is not contained in the context, say so plainly and suggest which page of "
    "the platform would have it (Risk Register, Geographic View, Forecasting, Model Governance, "
    "Case Management).\n"
    "3. Monetary figures in the context are in Albanian Lek (Lek), not euros or dollars. Always "
    "report them as Lek.\n"
    "4. Be concise: prefer short paragraphs or bullet points over long essays.\n"
    "5. Never state or imply that a customer is confirmed to be committing fraud. Risk scores and "
    "anomaly flags indicate a need for field verification, not proof of wrongdoing.\n"
    "6. Reply in the same language the question was asked in (Albanian or English)."
)


def build_platform_context(
    customer: pd.DataFrame,
    area: pd.DataFrame,
    priority: pd.DataFrame,
    cases: pd.DataFrame,
    metrics: dict,
    ingestion: dict,
    loss_forecast: pd.DataFrame,
    area_forecast: pd.DataFrame,
) -> str:
    """Summarize current pipeline outputs into a compact JSON the model can ground on.

    Kept intentionally small (a few dozen rows, not the full customer table) to stay
    well inside free-tier token limits and to keep the model's attention on what matters.
    """
    ctx: dict = {}

    if len(customer):
        ctx["totals"] = {
            "customers_analyzed": int(customer["customer_id"].nunique()),
            "high_or_critical": int(customer["risk_level"].isin(["High", "Critical"]).sum()),
            "critical": int((customer["risk_level"] == "Critical").sum()),
            "estimated_current_30d_loss_lek": round(float(pd.to_numeric(customer.get("estimated_loss_all_30d", 0), errors="coerce").fillna(0).sum()), 0),
        }
        top_cust_cols = [c for c in ["customer_id", "risk_score", "risk_level", "area_id", "transformer_id", "main_reason", "recommended_action", "estimated_loss_all_30d"] if c in customer.columns]
        ctx["top_20_customers_by_risk"] = customer.sort_values("risk_score", ascending=False).head(20)[top_cust_cols].to_dict(orient="records")

    if len(area):
        area_cols = [c for c in ["area_id", "area_risk_score", "area_risk_level", "customers", "high_risk_customers", "critical_customers", "anomaly_density", "estimated_loss_all_30d"] if c in area.columns]
        ctx["top_areas_by_risk"] = area.sort_values("area_risk_score", ascending=False).head(10)[area_cols].to_dict(orient="records")

    if len(priority):
        pcols = [c for c in ["priority_rank", "customer_id", "risk_score", "area_id", "main_reason"] if c in priority.columns]
        ctx["top_15_inspection_priority"] = priority.head(15)[pcols].to_dict(orient="records")

    if len(cases) and "status" in cases.columns:
        ctx["case_status_counts"] = cases["status"].value_counts().to_dict()

    if isinstance(metrics, dict) and metrics:
        ctx["model_metrics"] = {
            "expected_consumption_mae": metrics.get("expected_consumption", {}).get("expected_consumption_mae"),
            "fraud_classifier_roc_auc": metrics.get("models", {}).get("fraud_classifier_roc_auc"),
            "fraud_classifier_avg_precision": metrics.get("models", {}).get("fraud_classifier_avg_precision"),
            "note": "Metrics are measured on the currently loaded dataset, which may include sample/prototype customers.",
        }

    if isinstance(ingestion, dict) and ingestion:
        profile = ingestion.get("raw_profile", {})
        ctx["data_quality"] = {
            "detected_format": profile.get("detected_format"),
            "rows": profile.get("rows"),
            "columns": profile.get("columns"),
            "missing_percent": profile.get("missing_percent"),
            "schema_confidence": profile.get("schema_confidence"),
        }

    if len(loss_forecast):
        ctx["forecast_30d_total_loss_lek"] = round(float(pd.to_numeric(loss_forecast.get("predicted_loss_all", 0), errors="coerce").fillna(0).sum()), 0)
    if len(area_forecast):
        fcols = [c for c in ["area_id", "forecasted_area_risk_next_30d"] if c in area_forecast.columns]
        ctx["top_forecast_areas"] = area_forecast.head(5)[fcols].to_dict(orient="records") if fcols else []

    return json.dumps(ctx, default=str)
