# EnergyShield AI — Weather-Aware NTL Detection Platform

EnergyShield AI is a Streamlit-based decision-support platform for detecting and prioritizing **non-technical electricity losses (NTL)** such as suspicious consumption drops, meter tampering patterns, illegal-connection indicators, abnormal billing behavior, and geographic risk clusters.

The platform ingests smart-meter or billing datasets, cleans and standardizes them, engineers behavioral and contextual indicators, trains machine learning models, scores customers/areas/transformers, and provides an operational dashboard for analysts, administrators, and field inspectors.

> **Live Demo:** https://energyshield-ai-fuzjf6jth8cgs3rhyefh9r.streamlit.app/  
> **Repository:** mateossharka9-droid  
> **Status:** Prototype / hackathon demo  
> **Important:** EnergyShield is a risk-prioritization system, not legal proof of electricity theft. Final confirmation must come from field inspection and verified evidence.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Main Features](#main-features)
- [How the Platform Works](#how-the-platform-works)
- [Machine Learning Models](#machine-learning-models)
- [Risk Score Logic](#risk-score-logic)
- [Supported Dataset Formats](#supported-dataset-formats)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run the Platform Locally](#run-the-platform-locally)
- [Demo Accounts](#demo-accounts)
- [Generated Outputs](#generated-outputs)
- [Model Evaluation](#model-evaluation)
- [Gemini Operations Assistant](#gemini-operations-assistant)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

EnergyShield AI helps electricity distribution operators identify where non-technical losses may be happening and prioritize the most important inspection cases.

The platform combines:

- Historical consumption analysis
- Unsupervised anomaly detection
- Supervised fraud classification when labels exist
- Expected consumption prediction
- Weather-aware demand context
- Peer group comparison
- Geographic and transformer-level risk concentration
- Billing and payment behavior indicators
- Inspection priority ranking
- Forecasting of suspicious loss pressure
- Role-based dashboards for Admin, Analyst, and Inspector users

The goal is to support faster, more data-driven operational decisions for electricity loss reduction.

---

## Problem Statement

Electricity distribution companies lose revenue because of non-technical losses, including meter tampering, illegal connections, billing manipulation, and unregistered consumption. Traditional inspection methods are often manual, reactive, and expensive.

EnergyShield AI solves this by ranking customers and areas according to risk, so field teams can focus on the most suspicious and financially important cases first.

---

## Main Features

### 1. Data Intake and Cleaning

- Upload smart-meter, billing, or SGCC/STEG-style datasets.
- Automatically detect dataset schema.
- Convert wide-format meter readings into long analytical format.
- Clean invalid dates, missing values, duplicate records, negative readings, and extreme outliers.
- Produce an ingestion report and data-readiness score.

### 2. Customer Risk Scoring

Each customer receives:

- Final risk score from 0 to 100
- Risk level: Low, Medium, High, or Critical
- AI anomaly score
- Fraud probability when labels are available
- Estimated missing consumption
- Estimated 30-day revenue loss
- Main reason for the alert
- Recommended inspection action

### 3. Area and Transformer Analysis

The platform aggregates individual customer scores into:

- Area risk scores
- Transformer risk scores
- Geographic anomaly clusters
- Hotspot maps
- Inspection route files

### 4. Weather-Aware Analysis

EnergyShield uses weather context to reduce false positives. For example, high electricity usage during hot or cold days may be normal, while very low consumption during high-demand weather can be suspicious.

Weather features include:

- Mean temperature
- Minimum and maximum temperature
- Heating degree days
- Cooling degree days
- Weather demand pressure
- Cold / normal / hot classification
- Weather-consumption mismatch flags

### 5. Billing and Payment Risk

The platform can include operational payment indicators such as:

- Late payments in the last 12 months
- Unpaid bills
- Average payment delay
- Arrears amount
- Disconnections
- Months since last payment
- Account status
- Payment risk score

These are not treated as proof of fraud, but they are useful supporting indicators for inspection prioritization.

### 6. Forecasting

EnergyShield generates short-term forecasts for:

- Customer risk pressure
- Area risk pressure
- Estimated suspicious revenue loss
- Next 30-day operational priority

### 7. Role-Based Dashboard

The platform includes three operational roles:

- **Admin / Operations:** Upload data, run analysis, dispatch inspection duties, view all dashboards.
- **Data Analyst:** Review model quality, answer summary requests, inspect statistical results.
- **Field Inspector:** View assigned duties only, inspect customers, and report outcomes.

---

## How the Platform Works

```text
Raw electricity dataset
        ↓
Schema detection and format adaptation
        ↓
Data cleaning and standardization
        ↓
Weather, GIS, transformer, and billing context enrichment
        ↓
Customer-level feature engineering
        ↓
Expected consumption prediction
        ↓
Isolation Forest anomaly detection
        ↓
Optional supervised fraud classification
        ↓
Customer, area, and transformer risk scoring
        ↓
Inspection priority ranking
        ↓
Dashboard, map, reports, forecasts, and case management
```

The internal canonical format used by the platform is:

```text
customer_id | date | consumption_kwh | fraud_label(optional)
```

Additional columns such as `area_id`, `transformer_id`, `latitude`, `longitude`, `customer_type`, `meter_age`, and `contract_power_kw` are used when available.

---

## Machine Learning Models

EnergyShield uses standard machine learning algorithms from `scikit-learn`. The algorithms are not invented from scratch, but the models are trained/fitted inside the platform using the uploaded or prepared electricity dataset.

### 1. Expected Consumption Model

**Algorithm:** Random Forest Regressor  
**Saved model:** `models/expected_consumption_model.pkl`

This model predicts what a customer's normal recent consumption should be based on historical baseline, peer behavior, contract power, meter age, and weather context.

The platform compares expected consumption with actual consumption to calculate an expected-deviation signal.

### 2. Anomaly Detection Model

**Algorithm:** Isolation Forest  
**Saved model:** `models/isolation_forest_model.pkl`

This is an unsupervised model. It can detect suspicious behavior even when the dataset does not contain fraud labels.

It looks for unusual customers based on consumption behavior, peer deviation, weather mismatch, payment risk, and other engineered features.

### 3. Fraud Classification Model

**Algorithm:** Random Forest Classifier  
**Saved model:** `models/fraud_classifier.pkl`

This model is trained only when the dataset contains valid fraud labels. It predicts the probability that a customer belongs to the suspicious/fraud class.

If labels are not available, the platform still works using the unsupervised anomaly model and rule-based risk indicators.

---

## Risk Score Logic

The final customer risk score is a weighted combination of multiple signals:

```text
26% AI combined score
21% historical deviation
16% peer deviation
13% geographic risk
 8% sudden behavior flags
 8% weather context
 8% billing/payment behavior
```

Risk levels:

```text
0–30      Low
31–60     Medium
61–80     High
81–100    Critical
```

The platform also calculates an inspection priority score that combines risk, estimated loss value, and geographic concentration. This helps operations teams decide which cases should be inspected first.

---

## Supported Dataset Formats

### 1. SGCC Wide Smart-Meter Format

One row per customer and one column per day.

```text
UserId | IsStealer | 1/1/2014 | 1/2/2014 | 1/3/2014 | ...
```

or:

```text
CONS_NO | FLAG | 2014/1/1 | 2014/1/2 | 2014/1/3 | ...
```

### 2. Long Smart-Meter Format

```text
customer_id | date | consumption_kwh | fraud_label(optional)
```

Optional columns:

```text
area_id, transformer_id, latitude, longitude, city,
customer_type, meter_age, contract_power_kw
```

### 3. STEG Invoice/Billing Fraud ZIP

The platform can also process STEG-style billing fraud datasets:

```text
client_train.csv + invoice_train.csv
client_test.csv + invoice_test.csv
```

Invoices are converted into the internal customer/date/consumption format.

---

## Tech Stack

- **Python** — core programming language
- **Pandas / NumPy** — data cleaning and transformation
- **Scikit-learn** — machine learning models
- **Streamlit** — interactive web dashboard
- **Plotly** — charts and visual analytics
- **Folium / Streamlit-Folium** — maps and geospatial visualization
- **Joblib** — model saving/loading
- **OpenPyXL** — Excel support
- **Google Gemini API** — optional operations assistant

---

## Project Structure

```text
EnergyShield-NTL-Platform/
│
├── dashboard/
│   └── app.py                         # Main Streamlit dashboard
│
├── data/
│   └── raw/                           # Sample datasets and weather data
│
├── models/                            # Trained model files (.pkl)
│   ├── expected_consumption_model.pkl
│   ├── isolation_forest_model.pkl
│   └── fraud_classifier.pkl
│
├── outputs/                           # Generated analysis outputs
│
├── src/
│   ├── data_ingestion.py              # Schema detection and cleaning
│   ├── feature_engineering.py         # Feature extraction and indicators
│   ├── modeling.py                    # ML model training
│   ├── risk_scoring.py                # Final risk scores
│   ├── forecasting.py                 # Forecasting logic
│   ├── geospatial.py                  # Map and GIS outputs
│   ├── weather_features.py            # Weather context processing
│   ├── explainability.py              # Human-readable explanations
│   ├── gemini_assistant.py            # Optional AI assistant
│   └── pipeline.py                    # End-to-end pipeline orchestration
│
├── run_pipeline.py                    # CLI pipeline runner
├── requirements.txt                   # Python dependencies
├── MODEL_CARD.md                      # Model explanation and limitations
├── data_dictionary.md                 # Input/output field definitions
├── PLATFORM_GUIDE.md                  # Full platform documentation
└── README.md                          # Project overview
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Platform Locally

Run the data pipeline with the included sample dataset:

```bash
python run_pipeline.py --input data/raw/selected_smart_grid_theft_sample.csv --weather data/raw/albania_weather_daily_2014_2016.csv --max-customers 1200 --max-days 730
```

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in your terminal.

---

## Demo Accounts

All demo accounts use the same password:

```text
Password: oshee123
```

| Username | Role |
|---|---|
| `admin` | Administration & Operations |
| `analyst` | Data Analytics Office |
| `inspector1` | Field Inspection Team 1 |
| `inspector2` | Field Inspection Team 2 |
| `inspector3` | Field Inspection Team 3 |

---

## Generated Outputs

After running the pipeline, the platform creates output files such as:

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
```

These files are used by the dashboard for visualization, reports, maps, and inspection workflows.

---

## Model Evaluation

When fraud labels are available, the platform reports metrics such as:

- ROC AUC
- Average precision
- Precision at top 10 percent
- Recall at top 10 percent
- Holdout positive rate

For the expected consumption model, the platform reports:

- Mean Absolute Error (MAE)

These metrics are stored in:

```text
outputs/model_metrics.json
```

```

For Streamlit Cloud, add the key in **App Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.0-flash"
```

Do not commit real API keys to GitHub.

---

## Deployment

The easiest deployment option is **Streamlit Community Cloud**.

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to:

```text
dashboard/app.py
```

5. Deploy the app.

Optional Streamlit Cloud settings:

```text
Python version: 3.11 or newer
Main file: dashboard/app.py
```

---

## Important GitHub Security Note

Before pushing to GitHub, make sure these files are not committed:

```text
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
```

The repository should include `.streamlit/secrets.example.toml`, but not the real `secrets.toml` file.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
.DS_Store
```

You may also exclude generated outputs and trained models if you want a cleaner repository:

```gitignore
outputs/*.csv
outputs/*.json
outputs/*.geojson
models/*.pkl
```

---

## Limitations

- The platform is a prototype and should not be used as legal proof of fraud.
- Public datasets may not perfectly represent Albanian electricity distribution behavior.
- Synthetic GIS, customer metadata, weather, and payment fields should be replaced with real operational data in production.
- Model results depend heavily on input data quality.
- Final decisions should always involve human review and field verification.

---

## Future Improvements

- Connect to real OSHEE smart-meter and billing systems.
- Add real-time data streaming from meters.
- Improve authentication and access control.
- Add inspection feedback loops for continuous model retraining.
- Add SHAP explainability for model-level feature contribution.
- Add more advanced time-series forecasting models.
- Integrate real GIS feeder/transformer topology.
- Add audit logs and production-grade monitoring.

---

## Author

**Mateos Sharka**  
Artificial Intelligence Student  
GitHub: `@mateossharka9`  
Project: EnergyShield AI — Non-Technical Loss Detection Platform

---

## Acknowledgement

This project was built as a prototype for demonstrating how data science, machine learning, and operational analytics can support electricity-loss detection and inspection prioritization.
