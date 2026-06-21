# EnergyShield AI — Complete Platform Guide

A full, plain-language + technical walkthrough of how the EnergyShield non-technical-loss (NTL)
platform is built: the end-to-end workflow, every main feature, and exactly how the data analyst
side works (cleaning, feature extraction, model training, indicators, and Model Governance).

> EnergyShield is a **decision-support** prototype for OSHEE-style operations. Every score is a
> prioritization signal for field inspection — never legal proof of theft. Confirmation always
> happens in the field, and confirmed outcomes feed back into the model.

---

## 1. What the platform is

EnergyShield ingests electricity smart-meter / billing data, detects suspicious consumption and
billing behaviour with AI, scores every customer / area / transformer for risk, estimates the money
at risk, maps where to inspect, forecasts next-month loss pressure, and runs a live three-department
operations workflow.

### The three departments (roles)

The app opens with a **department sign-in**. Each role only sees the features it needs, and all three
work on the **same live data** (file-backed shared state with auto-refreshing panels), so actions flow
between them in near real time.

| Role | Login | Sees / does |
|------|-------|-------------|
| **Admin (OSHEE operations)** | `admin` | Plain-language briefing, revenue-loss indicator, dashboards, **uploads datasets & runs the pipeline**, **dispatches inspection duties**, **requests analyst summaries**. |
| **Data Analyst** | `analyst` | Statistical results, model quality (Model Governance), and **answers the admin's summary requests** in real time. |
| **Field Inspector** | `inspector1/2/3` | Receives **only the duties dispatched to their team**, an assigned-only field map, Customer 360, and **reports inspection outcomes** back to the admin. |

(Demo password for all accounts: `oshee123`.)

---

## 2. End-to-end workflow (step by step)

```
        ┌──────────────┐
  RAW   │ 1. INGESTION │  read file → detect schema → reshape to long format
  DATA  └──────┬───────┘
               ▼
        ┌──────────────┐
        │ 2. CLEANING  │  fix dates, dedupe, cap outliers, fill gaps, readiness score
        └──────┬───────┘
               ▼
        ┌──────────────────────┐
        │ 3. ENRICH METADATA   │  GIS coords, meter age, contract power, BILLING/PAYMENT profile
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ 4. WEATHER CONTEXT   │  attach daily temperature, heating/cooling degree days
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ 5. FEATURE ENGINEER  │  daily flags → per-customer indicators (consumption, peer,
        │                      │  weather, payment), streaks, trend slope, peer groups
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ 6. AI MODELS         │  (a) Expected-consumption regressor
        │                      │  (b) Isolation Forest anomaly model
        │                      │  (c) Supervised fraud classifier (if labels exist)
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ 7. RISK SCORING      │  customer → area → transformer scores, risk levels,
        │                      │  estimated Lek loss, inspection priority queue
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ 8. EXPLAIN + GEO +   │  human-readable reasons & actions, maps, 30-day forecast
        │    FORECAST          │
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ 9. OPERATIONS LOOP   │  Admin dispatches → Inspector reports → Analyst summarizes
        └──────────────────────┘
```

The whole technical pipeline (steps 1–8) is `src/pipeline.py::run_ntl_pipeline`. It runs automatically
the first time outputs are missing, when the admin uploads a new dataset in **Data Intake**, or as a
one-time auto-upgrade when the saved outputs predate the latest model. All results are saved as CSV /
JSON in `outputs/` and the dashboard reads from there.

---

## 3. Main features (by page)

**Admin**
- **Admin Console** — plain-language situation briefing, the **Revenue Loss Indicator** (expected vs.
  recorded revenue and the suspicious gap), one-click **duty dispatch** to a team, and **analyst summary
  requests** with replies.
- **Command Center** — overall NTL risk, suspicious customers, hotspots, and the full Revenue Loss
  Indicator (expected-vs-actual waterfall + top loss areas).
- **Data Intake** — upload SGCC wide files, STEG ZIPs, or long smart-meter data; automatic schema
  detection or manual column mapping; runs the pipeline.
- **Risk Register / Customer 360 / Geographic View / Weather Context / Building Risk Lab / Forecasting**.
- **Case Management** — turn alerts into cases, assign teams, track status, capture outcomes.
- **Model Governance** & **Reports** — validation metrics and exports.

**Analyst**
- **Analyst Workspace** — headline statistics, model-quality KPIs, answer admin summary requests live.
- **Model Governance** — full validation, correlation heatmap, probability analysis.

**Inspector**
- **My Inspections** — auto-refreshing queue of duties dispatched to *their* team; report outcomes.
- **Geographic View (assigned only)** — a map showing **only their dispatched customers** and the
  areas/transformers they must inspect.
- **Customer 360** — full investigation view per customer.

---

## 4. Data Analyst deep-dive (how it's actually built)

### 4.1 Ingestion & schema detection — `src/data_ingestion.py`
The platform accepts three real input shapes and auto-detects which one it is:

1. **SGCC wide** — one row per customer, one *column per day* (≥ 2 parseable date columns trigger this).
   It also handles messy variants where the ID/FLAG columns are in odd places.
2. **Long** — `customer_id | date | consumption_kwh | fraud_label(optional)`.
3. **STEG invoice ZIP** — client + invoice tables; invoices are converted to a daily-style series and
   consumption is normalized by `months_number`.

**Algorithm:**
- `detect_schema()` finds the customer, label, date, and consumption columns using candidate-name
  matching plus date-column-name parsing (`_parse_date_col_name`). It returns a **schema confidence
  score** and a list of data issues (e.g., "no label → unsupervised mode").
- `convert_to_long()` reshapes any supported schema into the canonical long format. Wide files are
  melted (one row per customer-day); if no customer ID is found, temporary `ROW_xxxxxx` IDs are created.
  `max_customers` / `max_days` let the analyst cap very large files.
- Manual mapping (`prepare_manual_mapping_dataset`) is the fallback when auto-detection isn't confident.

### 4.2 Cleaning — `clean_long_data()`
Deterministic, auditable cleaning steps (every count is reported in `ingestion_report.json`):
1. Trim/clean `customer_id`; drop blank IDs.
2. Parse `date` (`errors="coerce"`); drop rows with invalid customer/date.
3. Coerce `consumption_kwh` to numeric, including **European decimal-comma** handling (`_coerce_consumption`).
4. **Duplicate (customer, date) readings** → merged by **mean**; metadata kept from the first non-null.
5. **Negative readings** → set to NaN (then filled).
6. **Outlier capping** at the **99.5th percentile** to stop extreme spikes from distorting the model.
7. **Missing-value fill** ladder: customer median → global median → 0.
8. **Labels** (`fraud_label`) are propagated to customer level (SGCC repeats the label per day).
9. Sort by customer/date.

**Data readiness score (0–100)** = `0.30·completeness + 0.25·date_quality + 0.20·duplicate_quality +
0.25·history_quality` (history quality saturates at 90 days). This tells a non-technical user how
trustworthy the upload is before analysis.

### 4.3 Metadata & weather enrichment — `feature_engineering.add_missing_operational_metadata`, `weather_features.py`
If the upload lacks operational fields, the platform generates a realistic prototype set so the demo
always works (production note: replace with real GIS / meter registry / billing data):
- `customer_type`, `area_id`, `transformer_id`, `meter_age`, `contract_power_kw`, Albania GIS
  `latitude/longitude` (clustered around real city centres, sea-points filtered out).
- **Billing/payment profile** (the new fraud-trigger layer): `payment_late_count_12m`, `unpaid_bills`,
  `avg_payment_delay_days`, `arrears_amount_lek`, `disconnections_12m`, `months_since_last_payment`,
  `payment_method`, `account_status`. A latent "irregularity propensity" (Beta-distributed, raised for
  fraud-labelled customers) drives these so they are realistic and correlated — but not a perfect tell.
- **Weather**: daily temperature per city → `weather_class` (-1 cold / 0 normal / 1 hot),
  `weather_demand_pressure` (1 on cold/hot days), `heating_degree_days`, `cooling_degree_days`.

### 4.4 Feature extraction (the indicators) — `feature_engineering.py`
**Daily flags** (`build_daily_context`) per reading:
- Rolling baselines: `rolling_7d_mean`, `rolling_30d_mean`, `rolling_30d_std`, `previous_30d_mean` (shift 7).
- `drop_vs_previous_30d_pct`, `sudden_drop_flag` (drop > 60%), `sudden_spike_flag` (spike > 150%).
- `zero_flag` (≤ 0.05 kWh), `low_relative_flag` (< 25% of the customer's own median),
  `flatline_flag` (30-day std < 0.05 while mean > 0 → frozen/estimated readings).
- Weather mismatch: `low_on_extreme_weather_flag` (low usage on a hot/cold day),
  `high_on_normal_weather_flag`, combined into `weather_consumption_mismatch`.

**Per-customer features** (`build_customer_features`) aggregate the above into modelling indicators:
- Levels & volatility: `avg/median/std/total/max/min_consumption`, `cv_consumption`.
- Period comparison: `last_30_mean`, `previous_90_mean`, `first_90_mean`, `recent_drop_pct`, `recent_spike_pct`.
- Pattern ratios: `zero_day_ratio`, `low_day_ratio`, `flatline_ratio`, `sudden_drop_count`, `sudden_spike_count`.
- Streaks: `longest_zero_streak`, `longest_low_streak`, `longest_weather_mismatch_streak`.
- `consumption_trend_slope` — slope of a degree-1 line fit (`np.polyfit`) over the customer's series.
- **Peer comparison**: customers are grouped by `customer_type × consumption_band` (5 quantile bands via
  `qcut`); `peer_deviation_pct` and `customer_vs_peer_ratio` measure how far below similar profiles a
  customer sits.
- Composite rule scores (pre-AI): `sudden_behavior_score`, `weather_context_score`.
- `payment_risk_score` (0–100) = `0.30·late + 0.22·unpaid + 0.16·delay + 0.12·disconnections +
  0.10·arrears + 0.10·months_since` (+12 if account is under review/suspended), plus `payment_on_time_ratio`.

### 4.5 The three AI models — `src/modeling.py`
**(a) Expected-consumption model** — `RandomForestRegressor` (120 trees, `min_samples_leaf=3`, median
imputation). Trains on baseline/peer/weather features to predict each customer's *expected* recent
usage. The gap `expected_deviation_pct` (and the Lek loss later) comes from this. Quality = **MAE** on a
25% hold-out.

**(b) Isolation Forest (unsupervised anomaly detection)** — 250 trees, `contamination="auto"`, on the
full feature matrix (≈ 38 features incl. payment), standardized + imputed. The decision function is
inverted and normalized to a **0–100 `ai_anomaly_score`**; `isolation_forest_flag` marks the outliers.
This works even with **no labels at all**.

**(c) Supervised fraud classifier** — `RandomForestClassifier` (350 trees,
`class_weight="balanced_subsample"`, `min_samples_leaf=3`) trained **only if** confirmed labels exist
(≥ 2 classes, > 20 labelled). It outputs `fraud_probability` (0–100) and is validated on a stratified
25% hold-out (ROC AUC, average precision, precision/recall @ top-10%).

These are combined: `ai_combined_score = max(ai_anomaly_score, fraud_probability)` when a probability
exists, otherwise just the anomaly score.

### 4.6 Risk scoring — `src/risk_scoring.py`
1. **Preliminary customer risk** = `0.32·ai_combined + 0.28·historical + 0.18·peer + 0.14·sudden +
   0.08·weather`.
2. **Area & transformer risk** — aggregate preliminary scores; `anomaly_density` = share of high-risk
   customers; `area_risk_score = max(0.50·avg + 0.40·density·100 + 0.10·norm(critical), density·250)`
   so a tight geographic cluster stands out even when individual scores are moderate (transformer uses
   0.60/0.40 and ·230). `geographic_risk_score = 0.65·area + 0.35·transformer`.
3. **Final customer risk** (challenge weighting):
   `0.26·ai_combined + 0.21·historical + 0.16·peer + 0.13·geographic + 0.08·sudden + 0.08·weather +
   0.08·payment`. If a supervised probability exists, the final score is
   `max(base, 0.65·probability + 0.35·base)`.
4. **Risk levels:** `0–30 Low · 31–60 Medium · 61–80 High · 81–100 Critical`.

**Estimated money at risk** — `missing_kWh_30d = max(expected − actual, 0) · 30`;
`estimated_loss_all_30d = missing_kWh_30d × ERE tariff`, where the tariff uses the published Albanian
bands: **8.5 Lek/kWh** up to 700 kWh/month and **9.5 Lek/kWh** above (and for non-residential), pre-VAT.

**Inspection priority queue** — `0.55·risk + 0.25·norm(estimated_loss) + 0.20·geographic`, ranked,
top 200. This is what the admin dispatches from.

### 4.7 Explanations — `src/explainability.py`
Every customer gets a `main_reason`, a full `alert_explanation`, and a `recommended_action` written in
inspector language (drops, zeros, flatlines, peer gap, expected-vs-actual gap, weather mismatch, and
**payment irregularity**, e.g. "X late payments and Y unpaid bills, arrears … Lek").

### 4.8 Forecasting — `src/forecasting.py`
A robust, explainable forecaster (not a black box): for each daily signal it blends **same-weekday
seasonality + recent median level + a capped short trend**, clipped to recent operating ranges so it
never explodes or flatlines artificially. Daily suspicious-event pressure is converted to **money**
(anchored to the current monthly loss) and forecast directly, so the 30-day loss line follows the real
rising/falling trajectory. Outputs: a daily + **cumulative** loss forecast, a **next-30 vs last-30%**
comparison, an area next-period ranking, and a customer next-period risk ranking (now also influenced
by payment risk).

---

## 5. Indicator reference (quick lookup)

| Indicator | Meaning |
|-----------|---------|
| `risk_score` / `risk_level` | Final 0–100 suspicion score and its band. |
| `ai_anomaly_score` | Isolation-Forest outlierness (0–100), unsupervised. |
| `fraud_probability` | Supervised model's theft probability (0–100), only with labels. |
| `historical_deviation_score` | Drop vs own baseline + zero/low/flatline behaviour. |
| `peer_deviation_score` | How far below similar-profile customers. |
| `geographic_risk_score` | Blend of area + transformer concentration. |
| `sudden_flags_score` | Sudden drop/zero/low/flatline event intensity. |
| `weather_context_score` | Consumption that doesn't match cold/hot demand. |
| `payment_risk_score` | Billing discipline: late payments, unpaid bills, arrears, disconnections, dormancy. |
| `expected_deviation_pct` | Actual recent usage vs the regressor's expected usage. |
| `estimated_missing_kwh_30d` | Estimated under-recorded energy in 30 days. |
| `estimated_loss_all_30d` | That energy valued at the ERE/OSHEE tariff (Lek). |
| `inspection_priority_score` | Ranking used to build the dispatch queue. |
| `anomaly_density` | Share of high-risk customers in an area/transformer. |

---

## 6. Model Governance — what each item means

The **Model Governance** page is the analyst's validation cockpit. Top KPIs:

| Metric | What it measures | Good direction |
|--------|------------------|----------------|
| **Expected R²** | How well the expected-consumption regressor explains actual recent usage (1.0 = perfect). | Higher |
| **Expected MAE** | Average error of the expected-consumption model, in kWh. | Lower |
| **ROC AUC** | Supervised classifier's ability to rank fraud above non-fraud (0.5 = random, 1.0 = perfect). | Higher |
| **Avg Precision** | Area under the precision-recall curve — quality on the imbalanced positive (fraud) class. | Higher |
| **Precision@Top10%** | Of the top 10% highest-probability customers, the fraction that are truly fraud — i.e. **how clean the first inspection batch is**. | Higher |
| **Recall@Top10%** | Of *all* fraud cases, the fraction captured inside that top 10% — **how much theft you catch by inspecting only the top slice**. | Higher |
| **Features** | Number of model input features actually used on this dataset. | Context |

The three tabs:
- **Indicators** — headline statistics (mean/median consumption, volatility, mean risk, high/critical
  rates, total estimated 30-day loss) with a plain-language meaning per row.
- **Correlation heatmap** — how the risk drivers correlate with each other and with `fraud_probability`
  / `estimated_loss` (now including `payment_risk_score`, `unpaid_bills`, `arrears`). Use it to confirm
  signals behave sensibly (e.g., payment irregularity correlates positively with risk).
- **Probability analysis** — scatter of supervised probability vs final risk score, coloured by level,
  to see agreement/disagreement between the two views.

> Validation note shown on the page: these metrics are computed on the **currently loaded dataset** and
> are decision-support, not proof. For production, validate against a held-out slice of real metered
> data and confirmed inspection outcomes.

---

## 7. The live operations loop (real time)

- Shared state lives in small JSON/CSV files under `outputs/workspace/` (activity log, summary-request
  queue, data-freshness timestamp).
- Admin **dispatches** top-priority cases → the inspector's **My Inspections** board auto-refreshes
  (`st.fragment(run_every=…)`) and shows them within seconds.
- Inspector **reports an outcome** → it's logged and visible to the admin.
- Admin **requests a summary** → the analyst sees it instantly, sends a (data-drafted) reply, and the
  admin reads it under "Latest analyst replies".

---

## 8. Outputs produced (in `outputs/`)

`clean_long_consumption.csv`, `customer_risk_scores.csv`, `area_risk_scores.csv`,
`transformer_risk_scores.csv`, `inspection_priority.csv`, `daily_dashboard.csv`,
`loss_forecast.csv` / `area_forecast.csv` / `customer_forecast.csv`, `ingestion_report.json`,
`model_metrics.json`, `pipeline_summary.json`, plus saved models in `models/` and case/workspace files.

---

## 9. Honest limitations (production notes)

- GIS, payment profile, and weather are **synthetic prototype fields** unless you wire in real OSHEE
  GIS, billing/collection, and a real weather feed.
- Tariff is a 2-band simplification (no progressive billing, VAT, or business contracts).
- The supervised model only learns when **confirmed inspection outcomes** are fed back — this is the
  single most valuable real-world improvement (close the feedback loop from Case Management).
- All scores are **prioritization**, not proof; field verification is mandatory before any action.
