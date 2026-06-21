"""Human-readable explanations and field recommendations."""

from __future__ import annotations

import pandas as pd


def _pct(x: float) -> str:
    return f"{max(0, x) * 100:.0f}%"


def explain_customer(row: pd.Series) -> tuple[str, str, str]:
    """Return main_reason, full explanation, recommended action."""
    reasons = []

    if row.get("recent_drop_pct", 0) > 0.45:
        reasons.append(f"Recent consumption dropped {_pct(row.get('recent_drop_pct', 0))} compared with the customer's previous baseline.")
    if row.get("peer_deviation_pct", 0) > 0.35:
        reasons.append(f"Customer consumes {_pct(row.get('peer_deviation_pct', 0))} less than similar consumption profiles.")
    if row.get("zero_day_ratio", 0) > 0.08 or row.get("longest_zero_streak", 0) >= 5:
        reasons.append(f"Detected {int(row.get('zero_days', 0))} zero/near-zero consumption days, longest streak {int(row.get('longest_zero_streak', 0))} days.")
    if row.get("flatline_ratio", 0) > 0.08 or row.get("flatline_days", 0) >= 10:
        reasons.append(f"Detected flat or almost constant readings for {int(row.get('flatline_days', 0))} days.")
    if row.get("sudden_drop_count", 0) >= 5:
        reasons.append(f"Detected {int(row.get('sudden_drop_count', 0))} sudden daily drop events.")
    if row.get("ai_anomaly_score", 0) > 75:
        reasons.append(f"AI anomaly model assigned a high anomaly score of {row.get('ai_anomaly_score', 0):.1f}/100.")
    if row.get("area_risk_score", 0) > 65:
        reasons.append(f"Customer is located in a high-risk area with area risk {row.get('area_risk_score', 0):.1f}/100.")
    if row.get("transformer_risk_score", 0) > 65:
        reasons.append(f"Transformer zone shows concentrated anomalies with transformer risk {row.get('transformer_risk_score', 0):.1f}/100.")
    if row.get("expected_deviation_pct", 0) > 0.35:
        reasons.append(f"Actual recent consumption is {_pct(row.get('expected_deviation_pct', 0))} below expected consumption model output.")
    if row.get("weather_mismatch_ratio", 0) > 0.08 or row.get("weather_context_score", 0) > 55:
        reasons.append(f"Weather context mismatch detected: {int(row.get('weather_mismatch_days', 0))} days where consumption did not match cold/hot demand expectations.")
    payment_reason = None
    if row.get("payment_risk_score", 0) > 55 or row.get("unpaid_bills", 0) >= 3:
        arrears_val = float(row.get("arrears_amount_lek", 0) or 0)
        arrears_txt = f", arrears {arrears_val:,.0f} Lek" if arrears_val > 0 else ""
        payment_reason = (
            f"Irregular billing/payment behavior: {int(row.get('payment_late_count_12m', 0))} late payments and "
            f"{int(row.get('unpaid_bills', 0))} unpaid bills in 12 months{arrears_txt}."
        )
    elif row.get("disconnections_12m", 0) >= 1 and row.get("payment_risk_score", 0) > 35:
        payment_reason = f"{int(row.get('disconnections_12m', 0))} non-payment disconnection(s) on record — verify reconnection and meter integrity."
    if payment_reason:
        # Surface payment behavior prominently when it is a dominant signal.
        if row.get("payment_risk_score", 0) >= 70:
            reasons.insert(0, payment_reason)
        else:
            reasons.append(payment_reason)

    if not reasons:
        reasons.append("No severe suspicious pattern detected; customer behavior is close to historical and peer baseline.")

    # Action logic aligned to common NTL inspection checks.
    if row.get("zero_day_ratio", 0) > 0.08 or row.get("longest_zero_streak", 0) >= 5:
        action = "Verify physical meter reading, check disconnection/bypass possibility, inspect wiring and meter communication."
    elif row.get("flatline_ratio", 0) > 0.08:
        action = "Inspect meter integrity and data reporting; verify whether readings are frozen, estimated, or manipulated."
    elif row.get("recent_drop_pct", 0) > 0.45 and row.get("peer_deviation_pct", 0) > 0.35:
        action = "Inspect meter seal, wiring, and possible bypass; compare physical reading with billing system."
    elif row.get("weather_mismatch_ratio", 0) > 0.08:
        action = "Verify meter reading during recent cold/hot demand periods and compare with neighboring customers under the same weather conditions."
    elif row.get("area_risk_score", 0) > 65:
        action = "Prioritize field inspection together with nearby high-risk customers in the same area/transformer zone."
    elif row.get("payment_risk_score", 0) > 60 or row.get("unpaid_bills", 0) >= 4:
        action = "Combine collection and inspection: cross-check billing/payment history against metered consumption and verify the meter during a collection visit."
    else:
        action = "Monitor customer and schedule verification if the pattern continues."

    main_reason = reasons[0]
    full_explanation = " | ".join(reasons[:6])
    return main_reason, full_explanation, action


def add_explanations(customer_scores: pd.DataFrame) -> pd.DataFrame:
    df = customer_scores.copy()
    outputs = df.apply(explain_customer, axis=1, result_type="expand")
    outputs.columns = ["main_reason", "alert_explanation", "recommended_action"]
    return pd.concat([df, outputs], axis=1)
