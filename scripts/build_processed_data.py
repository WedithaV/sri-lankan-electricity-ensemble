"""Build the cleaned, small processed datasets used by the notebooks and the dashboard.

Outputs (data/processed/):
    monthly_clean.parquet          cleaned household-month panel with data-quality flags
    household_features_w1.parquet  one row per household, Wave 1 characteristics
    cleaning_log.json              what the cleaning rules changed

Run from the project root:  python scripts/build_processed_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_monthly_consumption  # noqa: E402
from src.data_processing import build_household_features_w1, clean_monthly  # noqa: E402

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

monthly, log = clean_monthly(load_monthly_consumption())
monthly.to_parquet(OUT / "monthly_clean.parquet", index=False)

hh = build_household_features_w1()
hh.to_parquet(OUT / "household_features_w1.parquet", index=False)

log["household_features_rows"] = int(len(hh))
log["household_features_columns"] = int(hh.shape[1] - 1)
(OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2))
print(json.dumps(log, indent=2))
