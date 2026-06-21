#!/usr/bin/env bash
set -e
python run_pipeline.py --input data/raw/selected_smart_grid_theft_sample.csv --weather data/raw/albania_weather_daily_2014_2016.csv --max-customers 1200 --max-days 730
streamlit run dashboard/app.py
