"""Loading functions for the RECON-SL dataset.

All paths are relative to the project root so the code works on Windows,
macOS/Linux and Streamlit Community Cloud.

Key data-quality decisions implemented here (see notebooks/01_dataset_inspection.ipynb):

* ``consumption`` in monthly_consumption.csv is stored as text. Values >= 1,000
  use a thousands separator (e.g. "1,159"). A naive ``pd.to_numeric`` would turn
  these into NaN and silently drop the highest-consuming households, so the
  comma is removed before conversion.
* Empty strings in ``consumption`` are genuine missing readings (590 rows) and
  are kept as NaN -- they are NOT converted to zero.
* ``month`` is stored as the last calendar day of the month (e.g. 2022-10-31);
  it is converted to a monthly Period/Timestamp for easy ordering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MONTHLY_PATH = RAW_DIR / "consumption_data" / "non_smart_meter" / "monthly_consumption.csv"
SURVEY_DIR = RAW_DIR / "survey_data"
SMART_15MIN_DIR = RAW_DIR / "consumption_data" / "smart_meter" / "15min_interval"
SMART_6HOUR_DIR = RAW_DIR / "consumption_data" / "smart_meter" / "6hour_interval"

# Unit of the monthly ``consumption`` column: energy in kilowatt-hours.
# Evidence (notebook 01, section 4):
#   * RECON-SL descriptor paper (Algama et al., arXiv:2609.28783): "Monthly consumption
#     data provides direct usage in kilowatt-hours".
#   * Billed values match smart-meter kWh register differences (median ratio 0.993).
# The data dictionary's wording "kilowatts (KW)" is a documentation discrepancy
# (kW is power, not energy).
CONSUMPTION_UNIT = "kWh"


def parse_consumption(values: pd.Series) -> pd.Series:
    """Convert the raw text consumption column to float.

    "1,159" -> 1159.0, "" -> NaN, "96" -> 96.0
    Raises ValueError if any non-empty value cannot be parsed, so that new
    unexpected formats are noticed instead of silently becoming NaN.
    """
    cleaned = values.astype("string").str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace("", pd.NA)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    unparsed = numeric.isna() & cleaned.notna()
    if unparsed.any():
        bad = cleaned[unparsed].unique()[:10]
        raise ValueError(f"Unparseable consumption values: {list(bad)}")
    return numeric.astype("float64")


def load_monthly_consumption(path: Path | str = MONTHLY_PATH) -> pd.DataFrame:
    """Load the monthly (billing) consumption panel.

    Returns columns: household_ID (str), month (Timestamp, first day of month),
    consumption (float, kWh, NaN where the reading is missing).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"household_ID", "month", "consumption"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"monthly_consumption.csv is missing columns: {missing_cols}")

    df["consumption"] = parse_consumption(df["consumption"])
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m-%d").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["household_ID", "month"]).reset_index(drop=True)
    return df


def load_survey_dates(path: Path | str = SURVEY_DIR / "survey_dates.csv") -> pd.DataFrame:
    """Load the per-household survey dates (dates are stored as MM/DD/YYYY)."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for col in ["wave1", "wave2", "wave3"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="raise")
    return df


def load_survey_table(wave: int, name: str) -> pd.DataFrame:
    """Load one survey table, e.g. ``load_survey_table(1, "household_information_and_history")``."""
    path = SURVEY_DIR / f"wave_{wave}" / f"w{wave}_{name}.csv"
    return pd.read_csv(path, low_memory=False)
