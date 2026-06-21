"""Geospatial outputs for maps and field inspection support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def valid_albania_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative coordinate validation for Albania-focused demo maps."""
    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        return df.copy()
    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    mask = out["latitude"].between(39.60, 42.75) & out["longitude"].between(18.80, 21.10)
    mask &= ~((out["longitude"] < 19.30) & (out["latitude"].between(39.70, 42.30)))
    return out[mask].copy()


def risk_color(score: float) -> str:
    if score >= 81:
        return "red"
    if score >= 61:
        return "orange"
    if score >= 31:
        return "yellow"
    return "green"


def export_customer_geojson(customer_scores: pd.DataFrame, output_path: str | Path, min_risk: float = 0) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = valid_albania_geo(customer_scores.copy())
    df = df[df["risk_score"] >= min_risk]
    features: List[Dict] = []
    for _, row in df.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
            "properties": {
                "customer_id": row.get("customer_id"),
                "risk_score": round(float(row.get("risk_score", 0)), 2),
                "risk_level": row.get("risk_level"),
                "area_id": row.get("area_id"),
                "transformer_id": row.get("transformer_id"),
                "main_reason": row.get("main_reason"),
                "recommended_action": row.get("recommended_action"),
                "color": risk_color(float(row.get("risk_score", 0))),
            },
        })
    geo = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    return output_path


def export_area_geojson(area_scores: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    area_scores = valid_albania_geo(area_scores.copy())
    features: List[Dict] = []
    for _, row in area_scores.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
            "properties": {
                "area_id": row.get("area_id"),
                "area_risk_score": round(float(row.get("area_risk_score", 0)), 2),
                "area_risk_level": row.get("area_risk_level"),
                "customers": int(row.get("customers", 0)),
                "high_risk_customers": int(row.get("high_risk_customers", 0)),
                "critical_customers": int(row.get("critical_customers", 0)),
                "anomaly_density": round(float(row.get("anomaly_density", 0)), 3),
                "color": risk_color(float(row.get("area_risk_score", 0))),
            },
        })
    geo = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    return output_path


def export_inspection_route(priority_df: pd.DataFrame, output_path: str | Path, max_points: int = 25) -> Path:
    """Create a simple ordered route from the inspection priority list.

    This is not road-routing. It is a priority-ordered planning route. In production,
    this should be replaced by a vehicle routing/road network optimizer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route = valid_albania_geo(priority_df).dropna(subset=["latitude", "longitude"]).head(max_points)
    coords = [[float(r["longitude"]), float(r["latitude"])] for _, r in route.iterrows()]
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"name": "Suggested inspection route", "points": len(coords)},
            }
        ] if len(coords) >= 2 else [],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    return output_path


def export_geospatial_outputs(customer_scores: pd.DataFrame, area_scores: pd.DataFrame, priority_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    export_customer_geojson(customer_scores, output_dir / "customer_risk_points.geojson", min_risk=30)
    export_area_geojson(area_scores, output_dir / "area_risk_points.geojson")
    export_inspection_route(priority_df, output_dir / "inspection_route.geojson")
