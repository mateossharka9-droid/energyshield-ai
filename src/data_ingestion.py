"""Robust dataset ingestion for EnergyShield NTL.

Purpose
-------
Utility and government datasets are rarely clean. This module accepts messy
smart-meter CSV/Excel/ZIP files and standardises them into the internal format:

    customer_id | date | consumption_kwh | fraud_label(optional)

Supported inputs
----------------
1. SGCC-style wide smart-meter data, even if the ID and FLAG columns are at
   the end of the file:
       01/01/2014 | 01/02/2014 | ... | CONS_NO | FLAG
       CONS_NO | 2014/1/1 | 2014/1/2 | ... | FLAG

2. Long smart-meter data:
       customer_id | date | consumption_kwh | fraud_label(optional)

3. CSV, Excel, or ZIP containing one CSV/Excel file.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

CUSTOMER_CANDIDATES = [
    "customer_id", "cust_id", "client_id", "consumer_id", "cons_no", "consno",
    "cons no", "contract_no", "contract", "account_no", "account", "subscriber_id",
    "meter_id", "meter_no", "id", "CONS_NO", "CONSNO", "ID",
]
DATE_CANDIDATES = ["date", "reading_date", "timestamp", "day", "datetime", "reading day"]
CONSUMPTION_CANDIDATES = [
    "consumption_kwh", "consumption", "kwh", "usage", "electricity_consumption",
    "energy", "reading", "meter_reading", "daily_consumption", "value",
]
LABEL_CANDIDATES = ["flag", "label", "fraud_label", "is_fraud", "target", "theft", "isstealer", "is_stealer", "stealer", "IsStealer", "FLAG", "Label"]


# STEG electricity/gas fraud dataset columns. This is an invoice/billing dataset,
# not daily smart-meter data, so it is adapted into the internal long format with
# invoice_date as the time index and normalized invoice consumption as kWh.
STEG_CLIENT_FILES = ("client_train.csv", "client_test.csv")
STEG_INVOICE_FILES = ("invoice_train.csv", "invoice_test.csv")
STEG_CONSUMPTION_COLUMNS = [
    "consommation_level_1", "consommation_level_2", "consommation_level_3", "consommation_level_4"
]

DATE_REGEXES = [
    # yyyy-mm-dd, yyyy/m/d
    r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*$",
    # mm/dd/yyyy or dd/mm/yyyy, also 1/13/2014
    r"^\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\s*$",
]


def _clean_header_value(value: object) -> str:
    """Normalise a column name while preserving its meaning."""
    s = str(value).replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _drop_empty_axes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    # Remove fully unnamed empty columns often created by Excel exports.
    keep_cols = []
    for c in df.columns:
        name = str(c).strip().lower()
        if name.startswith("unnamed") and df[c].isna().all():
            continue
        keep_cols.append(c)
    return df[keep_cols]


def _read_csv_robust(path_or_buffer) -> pd.DataFrame:
    """Read messy CSV files with delimiter/encoding fallback."""
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            # sep=None lets Python engine sniff comma/semicolon/tab.
            return pd.read_csv(
                path_or_buffer,
                sep=None,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
            try:
                if hasattr(path_or_buffer, "seek"):
                    path_or_buffer.seek(0)
            except Exception:
                pass
    raise last_error if last_error else ValueError("Could not read CSV file.")



def _zip_names(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return []
    with zipfile.ZipFile(path, "r") as zf:
        return zf.namelist()


def _is_steg_zip(path: str | Path) -> bool:
    """Return True when the ZIP looks like the STEG invoice fraud dataset."""
    try:
        names = {Path(n).name.lower() for n in _zip_names(path)}
    except Exception:
        return False
    return ("client_train.csv" in names and "invoice_train.csv" in names) or (
        "client_test.csv" in names and "invoice_test.csv" in names
    )


def _select_supported_file_from_zip(names: list[str]) -> str:
    """Pick the most useful tabular file from a ZIP.

    Avoid Kaggle sample-submission files and prefer training/all-data files.
    STEG multi-file ZIPs are handled separately by _read_steg_zip_as_long.
    """
    supported = [n for n in names if n.lower().endswith((".csv", ".xlsx", ".xls")) and not n.startswith("__MACOSX")]
    if not supported:
        raise ValueError("ZIP does not contain a CSV/XLSX file.")
    def score(name: str) -> tuple[int, str]:
        base = Path(name).name.lower()
        val = 0
        if "sample" in base or "submission" in base:
            val -= 100
        if "all" in base:
            val += 40
        if "train" in base:
            val += 30
        if "invoice" in base or "client" in base:
            val -= 10
        if base.endswith(".csv"):
            val += 5
        return (val, base)
    return sorted(supported, key=score, reverse=True)[0]


def _read_tabular(path: str | Path) -> pd.DataFrame:
    """Read CSV/XLSX or the best supported file inside ZIP.

    Multi-table fraud datasets such as STEG are intentionally not flattened here;
    ingest_dataset detects and adapts them using _read_steg_zip_as_long so that
    invoices and client labels can be joined correctly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".zip":
        if _is_steg_zip(path):
            raise ValueError(
                "Detected STEG invoice fraud ZIP. Use ingest_dataset(), which adapts the multi-file ZIP into long consumption format."
            )
        with zipfile.ZipFile(path, "r") as zf:
            name = _select_supported_file_from_zip(zf.namelist())
            with zf.open(name) as fh:
                if name.lower().endswith(".csv"):
                    df = _read_csv_robust(fh)
                else:
                    df = pd.read_excel(fh)
            return _drop_empty_axes(df)

    if suffix == ".csv":
        df = _read_csv_robust(path)
        return _drop_empty_axes(df)
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
        return _drop_empty_axes(df)
    raise ValueError("Unsupported file type. Please upload CSV, Excel, or ZIP.")


def _read_tabular_preview(path: str | Path, nrows: int = 200) -> pd.DataFrame:
    """Read only the first rows for dashboard inspection.

    This keeps large Kaggle/utility uploads responsive while still exposing
    headers, schema type, and a representative preview.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        if _is_steg_zip(path):
            raise ValueError("STEG ZIP preview is handled separately.")
        with zipfile.ZipFile(path, "r") as zf:
            name = _select_supported_file_from_zip(zf.namelist())
            with zf.open(name) as fh:
                if name.lower().endswith(".csv"):
                    return _drop_empty_axes(pd.read_csv(fh, nrows=nrows))
                return _drop_empty_axes(pd.read_excel(fh, nrows=nrows))
    if suffix == ".csv":
        return _drop_empty_axes(pd.read_csv(path, nrows=nrows))
    if suffix in [".xlsx", ".xls"]:
        return _drop_empty_axes(pd.read_excel(path, nrows=nrows))
    return _read_tabular(path).head(nrows)


def _read_csv_from_zip(zf: zipfile.ZipFile, file_name: str, **kwargs) -> pd.DataFrame:
    with zf.open(file_name) as fh:
        return pd.read_csv(fh, **kwargs)


def _find_zip_member(zf: zipfile.ZipFile, target_base: str) -> Optional[str]:
    target_base = target_base.lower()
    for name in zf.namelist():
        if Path(name).name.lower() == target_base:
            return name
    return None


def _choose_balanced_clients(clients: pd.DataFrame, max_customers: Optional[int], seed: int = 42) -> pd.DataFrame:
    """Sample clients without removing all fraud cases from imbalanced datasets."""
    if max_customers is None or max_customers <= 0 or len(clients) <= max_customers:
        return clients.copy()
    clients = clients.copy()
    label = pd.to_numeric(clients.get("target", pd.Series(index=clients.index, dtype=float)), errors="coerce")
    if label.notna().any() and label.nunique(dropna=True) >= 2:
        fraud = clients[label == 1]
        normal = clients[label != 1]
        n_fraud = min(len(fraud), max(1, int(max_customers * 0.35)))
        n_normal = max_customers - n_fraud
        parts = []
        if len(fraud):
            parts.append(fraud.sample(n=n_fraud, random_state=seed, replace=False))
        if len(normal):
            parts.append(normal.sample(n=min(n_normal, len(normal)), random_state=seed, replace=False))
        out = pd.concat(parts, ignore_index=True)
        if len(out) < max_customers:
            remaining = clients[~clients["client_id"].isin(out["client_id"])]
            if len(remaining):
                out = pd.concat([out, remaining.sample(n=min(max_customers-len(out), len(remaining)), random_state=seed)], ignore_index=True)
        return out
    return clients.sample(n=max_customers, random_state=seed).copy()


def _read_steg_zip_as_long(
    input_path: str | Path,
    max_customers: Optional[int] = None,
    max_days: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Adapt the STEG electricity/gas fraud dataset into internal long format.

    The STEG files contain customer metadata in client_train.csv and monthly/billing
    records in invoice_train.csv. For NTL screening, each invoice becomes one dated
    consumption observation. Consumption levels are summed and normalized by the
    invoice months_number so that customers with different billing cycle lengths are
    comparable.
    """
    input_path = Path(input_path)
    with zipfile.ZipFile(input_path, "r") as zf:
        client_name = _find_zip_member(zf, "client_train.csv") or _find_zip_member(zf, "client_test.csv")
        invoice_name = _find_zip_member(zf, "invoice_train.csv") or _find_zip_member(zf, "invoice_test.csv")
        if not client_name or not invoice_name:
            raise ValueError("STEG ZIP must contain client_train/client_test and invoice_train/invoice_test CSV files.")
        clients = _read_csv_from_zip(zf, client_name)
        clients.columns = [_clean_header_value(c) for c in clients.columns]
        if "client_id" not in clients.columns:
            raise ValueError("STEG client file is missing client_id.")
        clients = _choose_balanced_clients(clients, max_customers=max_customers, seed=seed)
        keep_ids = set(clients["client_id"].astype(str))

        chunks = []
        usecols = None
        # Read in chunks so the 300MB+ invoice file remains manageable.
        with zf.open(invoice_name) as fh:
            reader = pd.read_csv(fh, chunksize=250_000)
            for chunk in reader:
                chunk.columns = [_clean_header_value(c) for c in chunk.columns]
                if "client_id" not in chunk.columns:
                    raise ValueError("STEG invoice file is missing client_id.")
                chunk["client_id"] = chunk["client_id"].astype(str)
                chunk = chunk[chunk["client_id"].isin(keep_ids)].copy()
                if chunk.empty:
                    continue
                if "counter_type" in chunk.columns:
                    # Prioritize electricity records; keep all if field values are unexpected.
                    elec = chunk[chunk["counter_type"].astype(str).str.upper().eq("ELEC")]
                    if not elec.empty:
                        chunk = elec
                chunks.append(chunk)
        if not chunks:
            raise ValueError("No invoice records matched the selected STEG customers.")
        invoices = pd.concat(chunks, ignore_index=True)

    for c in STEG_CONSUMPTION_COLUMNS:
        if c not in invoices.columns:
            invoices[c] = 0
        invoices[c] = pd.to_numeric(invoices[c], errors="coerce").fillna(0)
    months = pd.to_numeric(invoices.get("months_number", 1), errors="coerce").replace(0, np.nan).fillna(1)
    coefficient = pd.to_numeric(invoices.get("counter_coefficient", 1), errors="coerce").replace(0, np.nan).fillna(1)
    invoices["consumption_kwh"] = invoices[STEG_CONSUMPTION_COLUMNS].sum(axis=1) * coefficient / months
    invoices["date"] = pd.to_datetime(invoices.get("invoice_date"), errors="coerce")
    if max_days is not None and max_days > 0:
        cutoff = invoices["date"].max() - pd.Timedelta(days=int(max_days))
        invoices = invoices[invoices["date"] >= cutoff].copy()

    clients = clients.rename(columns={"client_id": "customer_id", "target": "fraud_label"}).copy()
    invoices = invoices.rename(columns={"client_id": "customer_id"})
    merged = invoices.merge(clients, on="customer_id", how="left")

    long_df = pd.DataFrame({
        "customer_id": merged["customer_id"].astype(str),
        "date": merged["date"],
        "consumption_kwh": merged["consumption_kwh"],
        "fraud_label": pd.to_numeric(merged.get("fraud_label", np.nan), errors="coerce"),
    })
    if "disrict" in merged.columns:
        long_df["area_id"] = "DISTRICT_" + merged["disrict"].astype(str)
    elif "district" in merged.columns:
        long_df["area_id"] = "DISTRICT_" + merged["district"].astype(str)
    if "region" in merged.columns:
        long_df["transformer_id"] = "REGION_" + merged["region"].astype(str)
    if "client_catg" in merged.columns:
        long_df["customer_type"] = "category_" + merged["client_catg"].astype(str)
    if "creation_date" in merged.columns:
        creation = pd.to_datetime(merged["creation_date"], errors="coerce", dayfirst=True)
        long_df["meter_age"] = ((long_df["date"] - creation).dt.days / 365.25).clip(lower=0)

    adapter_report = {
        "detected_format": "steg_invoice_fraud",
        "source_files": [Path(input_path).name],
        "client_rows_loaded": int(len(clients)),
        "invoice_rows_loaded": int(len(invoices)),
        "note": "STEG invoice dataset adapted to customer_id/date/consumption_kwh/fraud_label using invoice_date and monthly normalized consumption.",
    }
    return long_df, adapter_report

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_header_value(c) for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    raw_lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", " ")
        if key in lookup:
            return lookup[key]
        if cand.lower() in raw_lookup:
            return raw_lookup[cand.lower()]
    return None


def _looks_like_date_string(s: str) -> bool:
    s = str(s).strip()
    return any(re.match(pattern, s) for pattern in DATE_REGEXES)


def _parse_date_col_name(col: object) -> Optional[pd.Timestamp]:
    """Return parsed date if a column name looks like a date, else None.

    Handles both SGCC-style `2014/1/1` and files like the user's sample:
    `01/01/2014`, `1/13/2014`.
    """
    s = _clean_header_value(col)
    if not _looks_like_date_string(s):
        return None
    for dayfirst in (False, True):
        try:
            dt = pd.to_datetime(s, errors="raise", dayfirst=dayfirst)
            if pd.isna(dt):
                continue
            # Protect against IDs accidentally parsed as dates.
            if 2000 <= int(dt.year) <= 2100:
                return pd.Timestamp(dt).normalize()
        except Exception:
            continue
    return None


def _infer_customer_col(df: pd.DataFrame, date_cols: list[str], label_col: Optional[str]) -> Optional[str]:
    """Infer customer ID column for messy wide files."""
    candidate_cols = [c for c in df.columns if c not in set(date_cols) and c != label_col]
    if not candidate_cols:
        return None
    # Prefer object-like columns with high uniqueness and values that look like IDs.
    scored = []
    for col in candidate_cols:
        s = df[col]
        non_null = s.dropna().astype(str).str.strip()
        if non_null.empty:
            continue
        unique_ratio = non_null.nunique() / max(len(non_null), 1)
        avg_len = non_null.str.len().mean()
        alpha_num_share = non_null.str.contains(r"[A-Za-z]", regex=True).mean()
        numeric_share = pd.to_numeric(non_null, errors="coerce").notna().mean()
        # Customer IDs often have high uniqueness, are not all numeric consumption values,
        # and may be hex/alphanumeric strings.
        score = unique_ratio * 2.0 + min(avg_len / 20, 1.0) + alpha_num_share - numeric_share * 0.25
        scored.append((score, col))
    if not scored:
        return candidate_cols[0]
    scored.sort(reverse=True)
    return scored[0][1]


def _schema_confidence(schema_type: str, customer_col: Optional[str], label_col: Optional[str], date_count: int, date_col: Optional[str], consumption_col: Optional[str]) -> float:
    score = 0.0
    if schema_type == "sgcc_wide":
        score += 0.45 if date_count >= 20 else 0.30 if date_count >= 5 else 0.0
        score += 0.30 if customer_col else 0.0
        score += 0.15 if label_col else 0.05
        score += 0.10
    elif schema_type == "long":
        score += 0.35 if customer_col else 0.0
        score += 0.25 if date_col else 0.0
        score += 0.25 if consumption_col else 0.0
        score += 0.15 if label_col else 0.05
    return round(float(min(score, 1.0)), 3)


def profile_raw_dataset(df: pd.DataFrame, schema: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Return a user-friendly data quality profile for the upload page."""
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    missing_pct = float(missing_cells / total_cells * 100) if total_cells else 0.0
    duplicate_rows = int(df.duplicated().sum())
    parsed_date_cols = [c for c in df.columns if _parse_date_col_name(c) is not None]
    numeric_cols = int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))
    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_percent": round(missing_pct, 2),
        "duplicate_rows": duplicate_rows,
        "detected_date_columns": int(len(parsed_date_cols)),
        "numeric_columns": numeric_cols,
    }
    if schema:
        profile.update({
            "detected_format": schema.get("schema_type"),
            "customer_column": schema.get("customer_col"),
            "label_column": schema.get("label_col"),
            "schema_confidence": schema.get("schema_confidence"),
        })
    return profile


def detect_schema(df: pd.DataFrame) -> Dict[str, object]:
    """Detect SGCC-wide or long smart-meter format, including messy variants."""
    df = _normalise_columns(df)
    customer_col = _find_col(df, CUSTOMER_CANDIDATES)
    label_col = _find_col(df, LABEL_CANDIDATES)
    date_col = _find_col(df, DATE_CANDIDATES)
    consumption_col = _find_col(df, CONSUMPTION_CANDIDATES)

    parsed_date_cols = []
    parsed_date_map = {}
    for c in df.columns:
        parsed = _parse_date_col_name(c)
        if parsed is not None:
            parsed_date_cols.append(c)
            parsed_date_map[c] = str(parsed.date())

    if len(parsed_date_cols) >= 2:
        schema_type = "sgcc_wide"
        if not customer_col:
            customer_col = _infer_customer_col(df, parsed_date_cols, label_col)
        if not customer_col:
            # Last fallback for extremely messy files: generate customer IDs from row numbers later.
            customer_col = None
    elif customer_col and date_col and consumption_col:
        schema_type = "long"
    else:
        raise ValueError(
            "Could not detect dataset schema. Expected SGCC wide data with date columns, "
            "long format with customer/date/consumption columns, or a STEG client/invoice ZIP. Open the Data Quality panel "
            "and map columns manually if needed."
        )

    confidence = _schema_confidence(schema_type, customer_col, label_col, len(parsed_date_cols), date_col, consumption_col)
    issues = []
    if schema_type == "sgcc_wide" and not customer_col:
        issues.append("Customer ID column was not detected; row numbers will be used as temporary IDs.")
    if schema_type == "sgcc_wide" and label_col is None:
        issues.append("No fraud/flag label detected; platform will run unsupervised anomaly detection.")
    if schema_type == "sgcc_wide" and len(parsed_date_cols) < 30:
        issues.append("Few date columns detected; risk scoring works better with longer consumption history.")
    if schema_type == "long" and label_col is None:
        issues.append("No fraud/flag label detected; platform will run unsupervised anomaly detection.")

    return {
        "schema_type": schema_type,
        "customer_col": customer_col,
        "label_col": label_col,
        "date_col": date_col,
        "consumption_col": consumption_col,
        "date_value_cols": parsed_date_cols,
        "date_column_map": parsed_date_map,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "schema_confidence": confidence,
        "issues": issues,
    }


def _coerce_consumption(series: pd.Series) -> pd.Series:
    """Convert messy consumption values into numeric kWh."""
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan, "-": np.nan})
    # Handle European decimal comma only when no normal decimal point exists.
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(r"(?<=\d),(?=\d{1,3}$)", ".", regex=True)
    return pd.to_numeric(s, errors="coerce")


def convert_to_long(
    df: pd.DataFrame,
    schema: Dict[str, object],
    max_customers: Optional[int] = None,
    max_days: Optional[int] = None,
) -> pd.DataFrame:
    """Convert supported schemas into customer_id/date/consumption_kwh/fraud_label."""
    df = _normalise_columns(df)
    schema_type = schema["schema_type"]
    customer_col = schema.get("customer_col")
    label_col = schema.get("label_col")

    if customer_col is None or customer_col not in df.columns:
        df = df.copy()
        customer_col = "Generated Customer ID"
        df[customer_col] = [f"ROW_{i+1:06d}" for i in range(len(df))]

    if max_customers is not None and max_customers > 0 and df[customer_col].nunique() > max_customers:
        keep = df[customer_col].drop_duplicates().sample(max_customers, random_state=42)
        df = df[df[customer_col].isin(keep)].copy()

    if schema_type == "sgcc_wide":
        date_cols = list(schema["date_value_cols"])
        date_map = schema.get("date_column_map", {})
        # Keep most recent max_days columns for speed and relevance.
        date_cols = sorted(date_cols, key=lambda c: pd.to_datetime(date_map.get(c, c), errors="coerce"))
        if max_days is not None and max_days > 0 and len(date_cols) > max_days:
            date_cols = date_cols[-max_days:]
        id_vars = [customer_col]
        if label_col and label_col in df.columns:
            id_vars.append(label_col)
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=date_cols,
            var_name="date",
            value_name="consumption_kwh",
        )
        long_df["date"] = long_df["date"].map(lambda c: date_map.get(c, c))
        long_df = long_df.rename(columns={customer_col: "customer_id"})
        if label_col and label_col in long_df.columns:
            long_df = long_df.rename(columns={label_col: "fraud_label"})
        else:
            long_df["fraud_label"] = np.nan

    elif schema_type == "long":
        long_df = df.rename(
            columns={
                customer_col: "customer_id",
                schema["date_col"]: "date",
                schema["consumption_col"]: "consumption_kwh",
            }
        ).copy()
        if label_col and label_col in long_df.columns:
            long_df = long_df.rename(columns={label_col: "fraud_label"})
        elif "fraud_label" not in long_df.columns:
            long_df["fraud_label"] = np.nan
        keep_cols = ["customer_id", "date", "consumption_kwh", "fraud_label"]
        keep_cols += [c for c in ["area_id", "transformer_id", "latitude", "longitude", "customer_type"] if c in long_df.columns]
        long_df = long_df[keep_cols]
    else:
        raise ValueError(f"Unsupported schema type: {schema_type}")

    return long_df


def clean_long_data(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Clean long-format smart-meter readings."""
    df = long_df.copy()
    initial_rows = len(df)

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = df[df["customer_id"].str.len() > 0].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["consumption_kwh"] = _coerce_consumption(df["consumption_kwh"])

    invalid_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["customer_id", "date"])

    duplicate_customer_date = int(df.duplicated(subset=["customer_id", "date"]).sum())
    if duplicate_customer_date:
        # If duplicate readings exist, use the mean reading for that date.
        label_cols = ["fraud_label"] if "fraud_label" in df.columns else []
        meta_cols = [c for c in ["area_id", "transformer_id", "latitude", "longitude", "customer_type"] if c in df.columns]
        agg = {"consumption_kwh": "mean"}
        for c in label_cols + meta_cols:
            agg[c] = lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
        df = df.groupby(["customer_id", "date"], as_index=False).agg(agg)

    negative_count = int((df["consumption_kwh"] < 0).sum())
    df.loc[df["consumption_kwh"] < 0, "consumption_kwh"] = np.nan

    missing_before = int(df["consumption_kwh"].isna().sum())
    non_na = df["consumption_kwh"].dropna()
    cap_value = float(non_na.quantile(0.995)) if len(non_na) else 0.0
    extreme_count = int((df["consumption_kwh"] > cap_value).sum()) if cap_value > 0 else 0
    if cap_value > 0:
        df["consumption_kwh"] = df["consumption_kwh"].clip(upper=cap_value)

    # Fill missing readings by customer median, then global median, then 0.
    customer_median = df.groupby("customer_id")["consumption_kwh"].transform("median")
    df["consumption_kwh"] = df["consumption_kwh"].fillna(customer_median)
    df["consumption_kwh"] = df["consumption_kwh"].fillna(df["consumption_kwh"].median()).fillna(0)

    if "fraud_label" in df.columns:
        df["fraud_label"] = pd.to_numeric(df["fraud_label"], errors="coerce")
        # Customer-level labels repeated daily in SGCC.
        df["fraud_label"] = df.groupby("customer_id")["fraud_label"].transform(
            lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
        )
    else:
        df["fraud_label"] = np.nan

    df = df.sort_values(["customer_id", "date"]).reset_index(drop=True)

    # Data readiness score for non-technical users.
    completeness = 1.0 - min(missing_before / max(initial_rows, 1), 1.0)
    date_quality = 1.0 - min(invalid_dates / max(initial_rows, 1), 1.0)
    duplicate_quality = 1.0 - min(duplicate_customer_date / max(initial_rows, 1), 1.0)
    history_days = df.groupby("customer_id")["date"].nunique().median() if len(df) else 0
    history_quality = min(float(history_days) / 90.0, 1.0)
    readiness = round(float((0.30 * completeness + 0.25 * date_quality + 0.20 * duplicate_quality + 0.25 * history_quality) * 100), 1)

    report = {
        "initial_rows": int(initial_rows),
        "clean_rows": int(len(df)),
        "customers": int(df["customer_id"].nunique()),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "invalid_dates_removed": invalid_dates,
        "duplicate_customer_date_readings_merged": duplicate_customer_date,
        "negative_values_replaced": negative_count,
        "missing_consumption_filled": missing_before,
        "extreme_values_capped": extreme_count,
        "cap_value_99_5_percentile": cap_value,
        "median_history_days_per_customer": float(history_days) if len(df) else 0.0,
        "data_readiness_score": readiness,
    }
    return df, report


def ingest_dataset(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    max_customers: Optional[int] = None,
    max_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Read, detect, reshape, clean, and save an ingestion report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if Path(input_path).suffix.lower() == ".zip" and _is_steg_zip(input_path):
        long_df, adapter_report = _read_steg_zip_as_long(input_path, max_customers=max_customers, max_days=max_days)
        schema = {
            "schema_type": "steg_invoice_fraud",
            "customer_col": "client_id",
            "label_col": "target",
            "date_col": "invoice_date",
            "consumption_col": "+".join(STEG_CONSUMPTION_COLUMNS),
            "schema_confidence": 0.92,
            "issues": [
                "STEG is invoice/billing data, not daily smart-meter data; consumption is normalized by months_number.",
                "Geographic fields are district/region codes unless real GIS coordinates are added."
            ],
        }
        raw_profile = {
            "rows": adapter_report.get("invoice_rows_loaded", 0),
            "columns": 4,
            "missing_cells": 0,
            "missing_percent": 0.0,
            "duplicate_rows": 0,
            "detected_date_columns": 1,
            "numeric_columns": 2,
            "detected_format": "steg_invoice_fraud",
            "customer_column": "client_id",
            "label_column": "target",
            "schema_confidence": 0.92,
        }
        clean_df, report = clean_long_data(long_df)
        full_report = {"raw_profile": raw_profile, "schema": schema, "adapter": adapter_report, "cleaning": report}
    else:
        raw_df = _read_tabular(input_path)
        raw_df = _normalise_columns(raw_df)
        schema = detect_schema(raw_df)
        raw_profile = profile_raw_dataset(raw_df, schema)
        long_df = convert_to_long(raw_df, schema, max_customers=max_customers, max_days=max_days)
        clean_df, report = clean_long_data(long_df)
        full_report = {"raw_profile": raw_profile, "schema": schema, "cleaning": report}

    with open(output_dir / "ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    clean_df.to_csv(output_dir / "clean_long_consumption.csv", index=False)
    return clean_df, full_report


def inspect_uploaded_dataset(input_path: str | Path) -> Dict[str, object]:
    """Inspect a dataset before running the full pipeline.

    Used by the dashboard upload page to show administrators what the platform
    detected and whether the file is ready for NTL analysis.
    """
    if Path(input_path).suffix.lower() == ".zip" and _is_steg_zip(input_path):
        with zipfile.ZipFile(input_path, "r") as zf:
            client_name = _find_zip_member(zf, "client_train.csv") or _find_zip_member(zf, "client_test.csv")
            invoice_name = _find_zip_member(zf, "invoice_train.csv") or _find_zip_member(zf, "invoice_test.csv")
            clients_preview = _read_csv_from_zip(zf, client_name, nrows=20) if client_name else pd.DataFrame()
            invoices_preview = _read_csv_from_zip(zf, invoice_name, nrows=20) if invoice_name else pd.DataFrame()
        preview = invoices_preview.merge(clients_preview, on="client_id", how="left") if "client_id" in invoices_preview.columns and "client_id" in clients_preview.columns else invoices_preview
        schema = {
            "schema_type": "steg_invoice_fraud",
            "customer_col": "client_id",
            "label_col": "target",
            "date_col": "invoice_date",
            "consumption_col": "+".join(STEG_CONSUMPTION_COLUMNS),
            "date_value_cols": [],
            "date_column_map": {},
            "rows": None,
            "columns": int(preview.shape[1]),
            "schema_confidence": 0.92,
            "issues": [
                "Detected STEG client/invoice fraud dataset. The pipeline will join client_train with invoice_train and normalize invoice consumption.",
                "This source supports fraud classification and risk scoring, but daily smart-meter pattern resolution is lower than SGCC."
            ],
        }
        profile = {
            "rows": "multi-file",
            "columns": int(preview.shape[1]),
            "missing_cells": int(preview.isna().sum().sum()) if len(preview) else 0,
            "missing_percent": round(float(preview.isna().sum().sum() / max(preview.shape[0]*preview.shape[1], 1) * 100), 2) if len(preview) else 0.0,
            "duplicate_rows": int(preview.duplicated().sum()) if len(preview) else 0,
            "detected_date_columns": 1,
            "numeric_columns": int(sum(pd.api.types.is_numeric_dtype(preview[c]) for c in preview.columns)) if len(preview) else 0,
            "detected_format": "steg_invoice_fraud",
            "customer_column": "client_id",
            "label_column": "target",
            "schema_confidence": 0.92,
        }
        return {"schema": schema, "profile": profile, "preview": preview.head(20)}

    raw_df = _normalise_columns(_read_tabular_preview(input_path, nrows=200))
    schema = detect_schema(raw_df)
    profile = profile_raw_dataset(raw_df, schema)
    profile["rows"] = f"preview {len(raw_df):,}+"
    preview = raw_df.head(20).copy()
    return {"schema": schema, "profile": profile, "preview": preview}


def prepare_manual_mapping_dataset(
    input_path: str | Path,
    output_path: str | Path,
    mapping: Dict[str, object],
    max_customers: Optional[int] = None,
    max_days: Optional[int] = None,
) -> Path:
    """Create a standardized long-format CSV from user-selected columns.

    This is used by the dashboard's Manual Mapping Wizard when automatic schema
    detection cannot fully understand a messy OSHEE/utility export.

    Output columns are compatible with the normal pipeline:
        customer_id | date | consumption_kwh | fraud_label
    """
    raw_df = _normalise_columns(_read_tabular(input_path))
    schema_type = str(mapping.get("schema_type", "wide")).lower()
    customer_col = mapping.get("customer_col")
    label_col = mapping.get("label_col")

    if not customer_col or customer_col not in raw_df.columns:
        raise ValueError("Manual mapping requires a valid customer ID column.")

    if max_customers is not None and max_customers > 0 and raw_df[customer_col].nunique() > max_customers:
        keep = raw_df[customer_col].drop_duplicates().sample(max_customers, random_state=42)
        raw_df = raw_df[raw_df[customer_col].isin(keep)].copy()

    if schema_type == "wide":
        date_cols = [c for c in mapping.get("date_cols", []) if c in raw_df.columns]
        if not date_cols:
            raise ValueError("Manual wide mapping requires at least one date reading column.")
        parsed = []
        for c in date_cols:
            dt = _parse_date_col_name(c)
            if dt is None:
                # Allow less standard names but try a generic parse.
                dt = pd.to_datetime(str(c), errors="coerce")
            if pd.isna(dt):
                continue
            parsed.append((c, pd.Timestamp(dt).normalize()))
        if not parsed:
            raise ValueError("None of the selected date columns could be parsed as dates.")
        parsed = sorted(parsed, key=lambda x: x[1])
        if max_days is not None and max_days > 0 and len(parsed) > max_days:
            parsed = parsed[-max_days:]
        date_cols = [c for c, _ in parsed]
        date_map = {c: str(dt.date()) for c, dt in parsed}
        id_vars = [customer_col]
        if label_col and label_col in raw_df.columns:
            id_vars.append(label_col)
        long_df = raw_df.melt(id_vars=id_vars, value_vars=date_cols, var_name="date", value_name="consumption_kwh")
        long_df["date"] = long_df["date"].map(date_map)
        long_df = long_df.rename(columns={customer_col: "customer_id"})
        if label_col and label_col in long_df.columns:
            long_df = long_df.rename(columns={label_col: "fraud_label"})
        else:
            long_df["fraud_label"] = np.nan
        long_df = long_df[["customer_id", "date", "consumption_kwh", "fraud_label"]]

    elif schema_type == "long":
        date_col = mapping.get("date_col")
        consumption_col = mapping.get("consumption_col")
        if not date_col or date_col not in raw_df.columns:
            raise ValueError("Manual long mapping requires a valid date column.")
        if not consumption_col or consumption_col not in raw_df.columns:
            raise ValueError("Manual long mapping requires a valid consumption column.")
        long_df = raw_df.rename(columns={customer_col: "customer_id", date_col: "date", consumption_col: "consumption_kwh"}).copy()
        if label_col and label_col in long_df.columns:
            long_df = long_df.rename(columns={label_col: "fraud_label"})
        else:
            long_df["fraud_label"] = np.nan
        long_df = long_df[["customer_id", "date", "consumption_kwh", "fraud_label"]]
    else:
        raise ValueError("Manual mapping schema_type must be 'wide' or 'long'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path, index=False)
    return output_path
