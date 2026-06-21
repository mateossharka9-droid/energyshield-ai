# EnergyShield AI Model Card

## Purpose
EnergyShield prioritizes customers and areas for non-technical loss (NTL) inspection. It is a decision-support system, not legal proof of electricity theft.

## Inputs
- Historical customer consumption readings
- Optional fraud/inspection labels
- Customer peer group and contract metadata
- Area/transformer/GIS context
- Albania weather context: cold / normal / hot, heating/cooling degree days, and weather-consumption mismatch

## Model approach
1. Clean and standardize input data into `customer_id`, `date`, `consumption_kwh`, and optional `fraud_label`.
2. Build behavior features: recent drop, spikes, zero readings, flatline periods, volatility, trend, expected consumption, and peer deviation.
3. Add contextual features: area anomaly density, transformer risk, and weather demand pressure.
4. Train an unsupervised Isolation Forest for anomaly scoring.
5. When labels exist, train a class-balanced Random Forest classifier to estimate fraud probability.
6. Combine AI score, historical deviation, peer deviation, geographic risk, sudden flags, and weather mismatch into a 0–100 risk score.

## Validation principles
- Prefer holdout validation and inspection-confirmed outcomes.
- Monitor ROC AUC and average precision when labels exist.
- Use Precision@Top10% and Recall@Top10% because inspection teams act on ranked queues.
- Track false positives and confirmed NTL rate after field verification.

## Limitations
- Weather context explains expected demand but does not prove fraud.
- Synthetic GIS/weather fields are acceptable for prototype testing but must be replaced with real OSHEE/GIS data in production.
- Fraud labels from public datasets may not match Albanian operational behavior exactly.
