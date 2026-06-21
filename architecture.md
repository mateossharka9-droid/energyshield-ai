# EnergyShield AI Architecture

## 1. Ingestion Layer

Files:

```text
src/data_ingestion.py
```

Responsibilities:

- read CSV, Excel, ZIP
- detect SGCC wide format
- detect long smart-meter format
- convert data into internal time-series schema
- clean missing/negative/extreme values
- save ingestion report

Internal schema:

```text
customer_id
date
consumption_kwh
fraud_label(optional)
```

## 2. Feature Engineering Layer

Files:

```text
src/feature_engineering.py
```

Features:

- rolling 7-day mean
- rolling 30-day mean
- previous 30-day mean
- recent drop percentage
- sudden drop flag
- sudden spike flag
- zero consumption flag
- low relative consumption flag
- flatline flag
- customer trend slope
- peer profile deviation
- area/transformer metadata

## 3. AI Layer

Files:

```text
src/modeling.py
```

Models:

- Random Forest expected consumption model
- Isolation Forest anomaly model
- Gradient Boosting fraud classifier when labels exist

Outputs:

- expected recent consumption
- AI anomaly score
- fraud probability

## 4. Risk Scoring Layer

Files:

```text
src/risk_scoring.py
```

Calculates:

- customer Risk Score
- area Risk Score
- transformer Risk Score
- estimated missing kWh
- estimated suspicious monthly loss
- inspection priority score

## 5. Explainability Layer

Files:

```text
src/explainability.py
```

Creates human-readable explanations:

- main reason
- full alert explanation
- recommended inspection action

## 6. Geospatial Layer

Files:

```text
src/geospatial.py
```

Exports:

- customer risk GeoJSON
- area risk GeoJSON
- inspection route GeoJSON

## 7. Dashboard Layer

Files:

```text
dashboard/app.py
```

Pages:

- Admin Upload
- Executive Overview
- Customer Risk List
- Customer Investigation
- Geospatial Hotspots
- Inspection Priority
- Model & Data Reports
