"""AI models for EnergyShield NTL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FEATURES = [
    "avg_consumption", "std_consumption", "cv_consumption", "last_30_mean", "previous_90_mean",
    "recent_drop_pct", "recent_spike_pct", "zero_day_ratio", "low_day_ratio", "flatline_ratio",
    "sudden_drop_count", "sudden_spike_count", "longest_zero_streak", "longest_low_streak",
    "consumption_trend_slope", "peer_deviation_pct", "customer_vs_peer_ratio", "meter_age",
    "contract_power_kw", "sudden_behavior_score",
    "avg_temp_mean", "avg_heating_degree_days", "avg_cooling_degree_days",
    "avg_weather_demand_pressure", "weather_mismatch_ratio", "extreme_weather_day_ratio",
    "weather_context_score", "low_extreme_weather_days", "high_normal_weather_days",
    "payment_late_count_12m", "unpaid_bills", "avg_payment_delay_days", "arrears_amount_lek",
    "disconnections_12m", "months_since_last_payment", "payment_on_time_ratio", "payment_risk_score",
]

EXPECTED_FEATURES = [
    "previous_90_mean", "first_90_mean", "avg_consumption", "std_consumption", "cv_consumption",
    "peer_avg_last_30", "contract_power_kw", "meter_age",
    "avg_heating_degree_days", "avg_cooling_degree_days", "avg_weather_demand_pressure",
]


def _safe_numeric_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for f in features:
        if f in df.columns:
            values = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() == 0:
                values = pd.Series(0.0, index=df.index)
            out[f] = values
        else:
            out[f] = 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def add_expected_consumption_model(customer_features: pd.DataFrame, model_dir: str | Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Predict expected recent consumption using customer baseline and peer context."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = customer_features.copy()
    X = _safe_numeric_frame(df, EXPECTED_FEATURES)
    y = pd.to_numeric(df["last_30_mean"], errors="coerce").fillna(0)

    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3, n_jobs=-1)),
    ])
    metrics: Dict[str, float] = {}
    if len(df) > 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        metrics["expected_consumption_mae"] = float(mean_absolute_error(y_test, pred))
    reg.fit(X, y)
    df["expected_last_30_consumption"] = np.maximum(reg.predict(X), 0.01)
    df["expected_deviation_pct"] = ((df["expected_last_30_consumption"] - df["last_30_mean"]) / df["expected_last_30_consumption"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    joblib.dump(reg, model_dir / "expected_consumption_model.pkl")
    return df, metrics


def fit_anomaly_model(customer_features: pd.DataFrame, model_dir: str | Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fit unsupervised Isolation Forest and optional supervised classifier."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = customer_features.copy()
    features = [f for f in MODEL_FEATURES + ["expected_deviation_pct"] if f in df.columns]
    X = _safe_numeric_frame(df, features)

    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1)),
    ])
    iso.fit(X)
    # Lower decision_function means more anomalous. Convert to 0-100 anomaly score.
    decision = iso.named_steps["model"].decision_function(
        iso.named_steps["scaler"].transform(iso.named_steps["imputer"].transform(X))
    )
    raw = -decision
    if raw.max() - raw.min() > 1e-9:
        anomaly_score = (raw - raw.min()) / (raw.max() - raw.min()) * 100
    else:
        anomaly_score = np.zeros(len(raw))
    df["ai_anomaly_score"] = anomaly_score
    df["isolation_forest_flag"] = (iso.predict(X) == -1).astype(int)
    joblib.dump(iso, model_dir / "isolation_forest_model.pkl")

    metrics: Dict[str, float] = {"n_model_features": float(len(features))}

    # Supervised fraud probability if labels exist and have both classes.
    if "fraud_label" in df.columns and df["fraud_label"].notna().sum() > 20:
        labels = pd.to_numeric(df["fraud_label"], errors="coerce")
        labeled_mask = labels.notna()
        y = labels[labeled_mask].astype(int)
        if y.nunique() >= 2:
            X_labeled = X.loc[labeled_mask]
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=350,
                    random_state=42,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                )),
            ])
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_labeled, y, test_size=0.25, random_state=42, stratify=y
                )
                clf.fit(X_train, y_train)
                proba_test = clf.predict_proba(X_test)[:, 1]
                metrics["fraud_classifier_roc_auc"] = float(roc_auc_score(y_test, proba_test))
                metrics["fraud_classifier_avg_precision"] = float(average_precision_score(y_test, proba_test))
                # Inspection teams usually act on the top ranked customers, so top-k
                # precision is more operational than plain accuracy.
                eval_frame = pd.DataFrame({"y": y_test.to_numpy(), "p": proba_test}).sort_values("p", ascending=False)
                k = max(1, int(np.ceil(len(eval_frame) * 0.10)))
                top = eval_frame.head(k)
                metrics["precision_at_top_10_percent"] = float(top["y"].mean())
                metrics["recall_at_top_10_percent"] = float(top["y"].sum() / max(eval_frame["y"].sum(), 1))
                metrics["holdout_positive_rate"] = float(eval_frame["y"].mean())
            except Exception:
                # Very small/imbalanced uploads can fail stratified split. Still fit full classifier.
                pass
            clf.fit(X_labeled, y)
            df["fraud_probability"] = clf.predict_proba(X)[:, 1] * 100
            joblib.dump(clf, model_dir / "fraud_classifier.pkl")
        else:
            df["fraud_probability"] = np.nan
    else:
        df["fraud_probability"] = np.nan

    return df, metrics


def save_metrics(metrics: Dict[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
