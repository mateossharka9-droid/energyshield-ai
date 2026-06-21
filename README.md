# EnergyShield AI — Weather-Aware NTL Detection Platform

EnergyShield AI is a professional prototype platform for detecting and prioritizing non-technical electricity losses (NTL): suspicious consumption, meter tampering signals, illegal-connection indicators, abnormal billing/reading behavior, and geographic anomaly clusters.

> **Live demo:** `<paste your Streamlit Cloud URL here>`
> **Source code:** `<paste your GitHub repository URL here>`
>
> See [Deploy online and submit](#deploy-online-and-submit-challenge-submission) for step-by-step publishing instructions.

It covers the challenge requirements:

- historical consumption analysis
- automatic anomaly detection
- customer, area, and transformer risk scores
- interactive dashboard visualization
- explanations for each alert
- map of high-anomaly areas
- inspection-priority workflow
- weather/context features that reduce false positives

## Core workflow

```text
Electricity dataset upload
        ↓
Schema detection/adaptation
        ↓
Cleaning and standardization
        ↓
Weather and operational context merge
        ↓
Customer behavior feature engineering
        ↓
Peer comparison + geographic concentration
        ↓
Isolation Forest anomaly model
        ↓
Optional class-balanced fraud classifier if labels exist
        ↓
Risk score + alert explanation + inspection action
        ↓
Dashboard, map, forecasts, reports, and case register
```

## Supported dataset formats

### 1. SGCC wide smart-meter data

```text
UserId | IsStealer | 1/1/2014 | 1/2/2014 | ...
CONS_NO | FLAG | 2014/1/1 | 2014/1/2 | ...
```

This is the preferred format for daily electricity theft/anomaly detection.

### 2. Long smart-meter data

```text
customer_id | date | consumption_kwh | fraud_label(optional)
```

Optional fields are used automatically when available:

```text
area_id, transformer_id, latitude, longitude, city,
customer_type, meter_age, contract_power_kw
```

### 3. STEG invoice/billing fraud ZIP

The platform supports the public STEG fraud dataset structure:

```text
client_train.csv + invoice_train.csv
client_test.csv + invoice_test.csv
```

It joins customers and invoices, converts invoice rows into `customer_id / date / consumption_kwh / fraud_label`, and normalizes invoice consumption by `months_number`. This source is useful for fraud classification and billing-pattern analysis, but SGCC-style data is stronger for daily smart-meter anomaly detection.

## Weather context

The project includes a generated Albania daily weather dataset:

```text
data/raw/albania_weather_daily_2014_2016.csv
```

Weather columns include:

```text
temp_mean, temp_min, temp_max, precipitation_mm,
weather_class, is_cold, is_hot, weather_demand_pressure,
heating_degree_days, cooling_degree_days
```

Weather classification:

```text
-1 = cold
 0 = normal
 1 = hot
```

The model does **not** treat weather as proof of fraud. It uses weather as context. For example, high consumption during hot/cold periods can be normal, while very low consumption during high-demand weather can increase inspection priority.

## Model design

EnergyShield uses a multi-signal decision-support model:

1. **Historical deviation** — drops, spikes, zero readings, flatlines, recent vs previous baseline.
2. **Peer comparison** — compares customers with similar profiles and consumption bands.
3. **Expected consumption model** — Random Forest regressor estimates expected recent usage.
4. **Unsupervised anomaly detection** — Isolation Forest finds unusual customer behavior.
5. **Supervised fraud probability** — class-balanced Random Forest classifier is trained when labels exist.
6. **Geographic concentration** — area and transformer anomaly density.
7. **Weather mismatch** — checks whether consumption behavior matches cold/hot demand expectations.
8. **Billing / payment behavior** — late payments, unpaid bills, arrears, average payment delay, non-payment disconnections, and dormant accounts. Chronic payment irregularity is a strong real-world NTL trigger and is fed into both the risk score and the supervised classifier.

Final risk score weighting:

```text
26% AI combined score
21% historical deviation
16% peer deviation
13% geographic risk
 8% sudden behavior flags
 8% weather context
 8% billing / payment behavior
```

Payment-behavior fields (`payment_late_count_12m`, `unpaid_bills`, `avg_payment_delay_days`,
`arrears_amount_lek`, `disconnections_12m`, `months_since_last_payment`, `payment_method`,
`account_status`) are read from the uploaded dataset when present, and otherwise generated as a
realistic billing/collection profile so the signal is always available in the prototype.

Risk levels:

```text
0–30      Low
31–60     Medium
61–80     High
81–100    Critical
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the included SGCC-style sample with Albania weather context:

```bash
python run_pipeline.py --input data/raw/selected_smart_grid_theft_sample.csv --weather data/raw/albania_weather_daily_2014_2016.csv --max-customers 1200 --max-days 730
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Sign in with a demo account (see [Roles and departments](#roles-and-departments-multi-user)) — for example `admin` / `oshee123`.

Windows shortcut:

```bat
run_demo.bat
```

Linux/macOS shortcut:

```bash
bash run_demo.sh
```

## Run with STEG invoice fraud ZIP

```bash
python run_pipeline.py --input "path/to/archive.zip" --max-customers 1000 --max-days 730
```

For the uploaded `archive (12).zip`, the platform detects the STEG structure automatically and adapts it into the internal format.

## Roles and departments (multi-user)

The platform opens with a **department sign in**. Each role sees only the features it needs, and the three departments work on the same live data, so changes flow between them in near real time (file-backed shared state with auto-refreshing panels).

Demo accounts (password `oshee123` for all):

```text
admin       Administration & Operations
analyst     Data Analytics Office
inspector1  Field Inspection — Team 1
inspector2  Field Inspection — Team 2
inspector3  Field Inspection — Team 3
```

| Role | Can do | Cannot do |
| --- | --- | --- |
| **Admin** (OSHEE Operations) | Upload datasets and run analysis, see all dashboards and indicators, **dispatch inspection duties** to field teams, **request a summary** from the analyst, manage cases, export reports | — |
| **Analyst** (Data Analytics Office) | See statistical results and model quality, **answer the admin's summary requests** (auto-drafted from the data, editable), review risk register / forecasts / governance | Upload data, dispatch duties |
| **Inspector** (Field Inspection Team) | See **only the duties dispatched to their team** (live), investigate the customer, and **report the outcome** back to admin | See other teams, upload data, change the model |

Real-time flow:

```text
Admin uploads data and runs analysis  →  scores update for everyone
Admin dispatches duties to a team      →  that team's board updates live
Inspector reports an outcome           →  appears in the admin live activity feed
Admin requests a summary               →  analyst inbox updates; analyst replies; admin sees the response
```

A live status strip (data freshness + pending requests + latest action) and a cross-department activity feed give the real-time experience without a page reload.

## Dashboard pages

**Admin**

- **Admin Console** — plain-language briefing (how many customers are suspicious, how much money is at risk, where to focus), the **Revenue Loss Indicator** headline (expected vs. actually-recorded revenue and the suspicious gap), one-click duty dispatch, analyst summary requests, and a live coordination feed.
- **Command Center** — overall NTL risk, suspicious customers, key hotspots, and the full **Revenue Loss Indicator**: an expected-vs-actual revenue waterfall plus the top loss-contributing areas, all valued at the ERE/OSHEE tariff bands.
- **Data Intake** — upload SGCC, STEG, or long smart-meter data and run the pipeline.
- **Risk Register / Customer 360 / Geographic View / Weather Context / Building Risk Lab / Forecasting** — investigation and planning views.
- **Case Management** — assign/track inspection cases and verification status.
- **Reports / Operations Assistant** — management exports and the AI assistant.

**Analyst**

- **Analyst Workspace** — key statistical fields and the request inbox to answer admin summaries in real time.
- **Model Governance** — validation metrics, top-k inspection quality, and correlation analysis.
- **Risk Register / Customer 360 / Weather Context / Forecasting / Building Risk Lab / Reports / Assistant** — read access for analysis.

**Inspector**

- **My Inspections** — live board of duties dispatched to the inspector's team, with the reason, location, recommended check, and an outcome-reporting form.
- **Customer 360 / Geographic View** — context for the assigned customer and its location.

## Output files

```text
outputs/customer_risk_scores.csv
outputs/area_risk_scores.csv
outputs/transformer_risk_scores.csv
outputs/inspection_priority.csv
outputs/daily_dashboard.csv
outputs/customer_risk_points.geojson
outputs/area_risk_points.geojson
outputs/inspection_route.geojson
outputs/ingestion_report.json
outputs/model_metrics.json
outputs/pipeline_summary.json
outputs/workspace/activity_log.json      # cross-department live activity
outputs/workspace/summary_requests.json  # admin -> analyst request/response queue
```

## API key / assistant behavior

The dashboard does **not** require a Gemini key. If no key exists, the Operations Assistant uses a built-in operational analyst mode instead of crashing.

To enable Gemini later, set one of these:

```bash
set GEMINI_API_KEY=your_key_here      # Windows CMD
$env:GEMINI_API_KEY="your_key_here"   # PowerShell
export GEMINI_API_KEY=your_key_here   # macOS/Linux
```

or create:

```text
.streamlit/secrets.toml
```

using `.streamlit/secrets.example.toml` as the template.

## Production notes

For a real electricity distribution operator, replace prototype fields with:

- real AMI/smart-meter readings
- meter registry and customer contract data
- transformer topology and GIS coordinates
- inspection outcomes and confirmed NTL labels
- real historical weather feed from Open-Meteo, Meteostat, NASA POWER, or national meteorological stations
- role-based access control and audit logs

EnergyShield is a decision-support system. Field inspection and verified evidence remain the final confirmation step.

## Building / Village Risk Lab

This version includes an independent sidebar module named **Building Risk Lab**. It does not change the main SGCC/customer NTL pipeline.

The module can:

- Generate an Albania building/apartment/village consumption dataset with floors, unit metadata, city/village context, meter type, weather context, and hidden fraud patterns.
- Upload CSV/XLSX/ZIP building datasets.
- Score risk per apartment/unit, floor, and building.
- Estimate fraud probability and 30-day suspicious loss.
- Visualize building consumption against expected demand.
- Show a heatmap and marker map of high-risk buildings.
- Export unit scores, building scores, floor scores, and an operational report.

Recommended upload schema:

```text
date, city, location_type, building_id, building_type, floor, unit_id, unit_type,
household_size, area_sqm, meter_type, connection_type, contracted_power_kw,
latitude, longitude, expected_kwh, consumption_kwh, fraud_label
```

Only these fields are mandatory:

```text
date, building_id, unit_id or customer_id, consumption_kwh
```

The scoring approach combines expected-consumption deviation, peer comparison, meter behavior, weather-consumption mismatch, AI outlier detection, estimated loss impact, and supervised probability when labels exist.

## Guided start (optional)

When the platform opens for the first time it shows a short, optional **Guided Start**. The worker picks a role (Distribution Operator, Field Inspector, or Analytical/Operational Team) and what they want to do first, and the platform opens the matching workspace. It can always be skipped with **Skip and explore on my own**, and reopened anytime from the sidebar **Guided start** button. The guided start only changes which page opens — it never changes the data or the AI model.

## Deploy online and submit (challenge submission)

The submission needs two links: a **Project Demo URL** (a place where judges can use the live app) and a **Source Code URL** (where judges can read the code). The fastest free way to get both is GitHub + Streamlit Community Cloud.

### 1. Publish the source code on GitHub

Use the `EnergyShield-NTL-Platform` folder as the repository root so `requirements.txt` sits at the top level.

```bash
cd EnergyShield-NTL-Platform
git init
git add .
git commit -m "EnergyShield AI — NTL detection platform"
git branch -M main
git remote add origin https://github.com/<your-username>/energyshield-ntl.git
git push -u origin main
```

Make the repository **public** so judges can open it. That GitHub URL is your **Source Code URL**.

### 2. Deploy the live demo on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/energyshield-ntl`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **Python version (Advanced settings):** 3.11, 3.12, or 3.13 all work.
4. Click **Deploy**. The first load runs the pipeline on the bundled SGCC sample (about a minute), then the dashboard appears.

The resulting `https://<your-app>.streamlit.app` link is your **Project Demo URL**. It is public and shareable — anyone with the link can open it.

> Optional: to enable the Gemini Operations Assistant, open the app's **Settings → Secrets** on Streamlit Cloud and add:
> ```toml
> GEMINI_API_KEY = "your_key_here"
> GEMINI_MODEL = "gemini-2.0-flash"
> ```
> Without a key, the assistant automatically falls back to the built-in operational analyst, so the demo always works.

### 3. Put both links in your submission

Paste the two URLs into the challenge submission form (and at the top of this README):

- **Project Demo:** your `*.streamlit.app` link
- **Source Code:** your GitHub repository link

### Alternative free hosts

- **Hugging Face Spaces** — create a *Streamlit* Space, push the same files, set the app file to `dashboard/app.py`.
- **Render / Railway** — use the start command `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`.
