# Data Dictionary

## Input fields

| Field | Required | Description |
|---|---:|---|
| customer_id / CONS_NO | Yes | Customer or meter identifier |
| date columns or date | Yes | Reading date |
| consumption_kwh | Yes | Electricity consumption in kWh |
| flag / fraud_label | Optional | Known theft/fraud label, if available |
| area_id | Optional | Operational/geographic area |
| transformer_id | Optional | Transformer/feeder zone |
| latitude | Optional | Customer latitude |
| longitude | Optional | Customer longitude |
| customer_type | Optional | Residential/business/industrial |
| meter_age | Optional | Age of meter in years |
| contract_power_kw | Optional | Contracted power |

## Important generated features

| Feature | Meaning |
|---|---|
| rolling_7d_mean | Short-term consumption baseline |
| rolling_30d_mean | Monthly consumption baseline |
| sudden_drop_flag | Daily reading much lower than previous baseline |
| zero_flag | Zero or near-zero consumption |
| low_relative_flag | Consumption very low compared with customer's median |
| flatline_flag | Almost constant readings over time |
| recent_drop_pct | Recent consumption drop vs previous period |
| peer_deviation_pct | Difference from similar customer group |
| ai_anomaly_score | Isolation Forest anomaly score |
| fraud_probability | Supervised fraud probability if labels exist |
| area_risk_score | Geographic area risk |
| transformer_risk_score | Transformer/zone risk |
| risk_score | Final 0-100 customer risk |
| risk_level | Low, Medium, High, Critical |
| alert_explanation | Human-readable alert factors |
| recommended_action | Field inspection recommendation |
