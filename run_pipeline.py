from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_ntl_pipeline
from src.synthetic_generator import save_sgcc_wide_sample


def main():
    parser = argparse.ArgumentParser(description="Run EnergyShield AI NTL pipeline")
    parser.add_argument("--input", type=str, default="data/raw/selected_smart_grid_theft_sample.csv", help="CSV/XLSX/ZIP dataset path")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--max-customers", type=int, default=None, help="Optional customer sampling for large datasets")
    parser.add_argument("--max-days", type=int, default=None, help="Optional most recent days limit for large wide datasets")
    parser.add_argument("--make-sample", action="store_true", help="Generate a small SGCC-style sample file before running")
    parser.add_argument("--weather", type=str, default=None, help="Optional Albania daily weather CSV to merge by date + city")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.make_sample:
        print(f"Creating SGCC-style sample dataset at {input_path}")
        save_sgcc_wide_sample(input_path, n_customers=500, n_days=240)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}. Provide --input or run with --make-sample.")

    summary = run_ntl_pipeline(
        input_path=input_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_customers=args.max_customers,
        max_days=args.max_days,
        weather_path=args.weather,
    )
    print("\nEnergyShield AI pipeline completed.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
