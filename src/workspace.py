"""Shared, file-based collaboration state for the multi-department OSHEE workspace.

EnergyShield AI is operated by three departments that work on the same live data:

- **admin**     (OSHEE Operations): uploads datasets, reviews dashboards and indicators,
                dispatches inspection duties to field teams, and requests summaries from the
                data-analytics office.
- **analyst**   (Data Analytics Office): produces statistical results and answers the
                summary requests sent by the admin.
- **inspector** (Field Inspection Teams): receive dispatched duties and report verification
                outcomes back from the field.

State is stored as small JSON/CSV files inside ``outputs/workspace`` so that every logged-in
session reads and writes the same live information. This gives a near real-time experience as
data flows between departments. It is intentionally lightweight for the prototype; a production
deployment should replace it with a database and a real identity provider.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Users and roles (demo directory)
# -----------------------------------------------------------------------------
# Prototype credentials. In production these are replaced by an OSHEE identity provider.
USERS = {
    "admin": {"password": "oshee123", "role": "admin", "name": "OSHEE Operations Lead", "team": None},
    "analyst": {"password": "oshee123", "role": "analyst", "name": "Data Analytics Office", "team": None},
    "inspector1": {"password": "oshee123", "role": "inspector", "name": "Field Inspector — Team 1", "team": "Inspection Team 1"},
    "inspector2": {"password": "oshee123", "role": "inspector", "name": "Field Inspector — Team 2", "team": "Inspection Team 2"},
    "inspector3": {"password": "oshee123", "role": "inspector", "name": "Field Inspector — Team 3", "team": "Inspection Team 3"},
}

ROLE_LABELS = {
    "admin": "Administration & Operations",
    "analyst": "Data Analytics Office",
    "inspector": "Field Inspection Team",
}

SUMMARY_TOPICS = [
    "Overall NTL risk summary",
    "Highest-risk areas / hotspots",
    "Top customers to inspect first",
    "Estimated financial loss breakdown",
    "Model quality and reliability",
    "Data quality of the latest upload",
    "Weather-driven consumption anomalies",
]


def authenticate(username: str, password: str) -> Optional[dict]:
    """Return a user profile (without the password) on success, else None."""
    record = USERS.get((username or "").strip().lower())
    if not record:
        return None
    if str(password) != str(record["password"]):
        return None
    return {
        "username": (username or "").strip().lower(),
        "role": record["role"],
        "name": record["name"],
        "team": record["team"],
        "role_label": ROLE_LABELS.get(record["role"], record["role"].title()),
    }


def inspector_teams() -> list[str]:
    return sorted({u["team"] for u in USERS.values() if u.get("team")})


# -----------------------------------------------------------------------------
# Shared state files
# -----------------------------------------------------------------------------
def workspace_dir(output_dir: str | Path) -> Path:
    d = Path(output_dir) / "workspace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _activity_path(output_dir: str | Path) -> Path:
    return workspace_dir(output_dir) / "activity_log.json"


def _requests_path(output_dir: str | Path) -> Path:
    return workspace_dir(output_dir) / "summary_requests.json"


def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_json_list(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------------------------------
# Activity feed (cross-department, near real-time)
# -----------------------------------------------------------------------------
def log_activity(output_dir: str | Path, actor: str, role: str, action: str, detail: str = "") -> None:
    path = _activity_path(output_dir)
    events = _read_json_list(path)
    events.append({
        "time": _now(),
        "actor": actor,
        "role": ROLE_LABELS.get(role, role),
        "action": action,
        "detail": detail,
    })
    # Keep the feed bounded.
    events = events[-300:]
    _write_json_list(path, events)


def load_activity(output_dir: str | Path, limit: int = 50) -> pd.DataFrame:
    events = _read_json_list(_activity_path(output_dir))
    if not events:
        return pd.DataFrame(columns=["time", "actor", "role", "action", "detail"])
    df = pd.DataFrame(events)
    return df.iloc[::-1].head(limit).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Admin -> Analyst summary requests (request / response queue)
# -----------------------------------------------------------------------------
def add_summary_request(output_dir: str | Path, requested_by: str, topic: str, note: str = "") -> str:
    path = _requests_path(output_dir)
    items = _read_json_list(path)
    request_id = f"REQ-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
    items.append({
        "request_id": request_id,
        "requested_by": requested_by,
        "topic": topic,
        "note": note,
        "status": "Pending",
        "requested_at": _now(),
        "answered_by": "",
        "answered_at": "",
        "response": "",
    })
    _write_json_list(path, items)
    return request_id


def load_summary_requests(output_dir: str | Path) -> pd.DataFrame:
    items = _read_json_list(_requests_path(output_dir))
    cols = ["request_id", "requested_by", "topic", "note", "status", "requested_at", "answered_by", "answered_at", "response"]
    if not items:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(items)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].iloc[::-1].reset_index(drop=True)


def answer_summary_request(output_dir: str | Path, request_id: str, answered_by: str, response: str) -> bool:
    path = _requests_path(output_dir)
    items = _read_json_list(path)
    changed = False
    for item in items:
        if item.get("request_id") == request_id:
            item["status"] = "Answered"
            item["answered_by"] = answered_by
            item["answered_at"] = _now()
            item["response"] = response
            changed = True
            break
    if changed:
        _write_json_list(path, items)
    return changed


def pending_request_count(output_dir: str | Path) -> int:
    items = _read_json_list(_requests_path(output_dir))
    return sum(1 for i in items if i.get("status") != "Answered")


# -----------------------------------------------------------------------------
# Data freshness
# -----------------------------------------------------------------------------
def data_last_updated(output_dir: str | Path) -> Optional[str]:
    path = Path(output_dir) / "customer_risk_scores.csv"
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
