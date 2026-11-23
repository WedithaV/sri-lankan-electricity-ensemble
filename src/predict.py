"""Prediction for one household from its bill history (used by the Streamlit dashboard).

Training/serving consistency: the input history is turned into features with the SAME
functions used in training (src.feature_engineering), and the saved Pipeline applies the
SAME fitted preprocessing. Nothing is re-implemented here.

The model predicts the bill of the month immediately AFTER the last month provided
(one-month-ahead). It needs the household's own recent bills: it is designed for existing
customers with a billing history, not for brand-new households.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import (SURVEY_CATEGORICAL, SURVEY_NUMERIC, add_calendar_features,
                                     add_history_features, add_survey_features)
from src.preprocessing import model_input_columns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models" / "best_model.joblib"
MAX_PLAUSIBLE_KWH = 10_000.0   # far above the largest monthly bill in the data (3,025 kWh)


class InputError(ValueError):
    """Raised for invalid user input (shown as a friendly message in the dashboard)."""


def load_bundle(path: Path | str = DEFAULT_BUNDLE) -> dict:
    bundle = joblib.load(path)
    for key in ("pipeline", "feature_set", "metadata"):
        if key not in bundle:
            raise ValueError(f"Model bundle is missing '{key}'")
    return bundle


def validate_history(history: dict) -> pd.Series:
    """history: {month (any date-like): kWh or None}. Returns a monthly Series (NaN = missing bill)."""
    if not history:
        raise InputError("Enter at least the previous month's bill.")
    s = pd.Series({pd.Timestamp(k).to_period("M").to_timestamp(): v for k, v in history.items()}, dtype="float64")
    s = s.sort_index()
    if s.notna().sum() == 0:
        raise InputError("At least one previous bill must be entered.")
    if (s.dropna() < 0).any():
        raise InputError("Bills cannot be negative.")
    if (s.dropna() > MAX_PLAUSIBLE_KWH).any():
        raise InputError(f"A bill above {MAX_PLAUSIBLE_KWH:,.0f} kWh is not plausible for a household.")
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full)     # gaps become NaN (missing bills), never zero


def build_feature_row(history: dict, survey: dict | None = None, survey_date=None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Feature row for the month after the last bill in ``history``.

    ``survey`` (optional): dict of Wave-1-style household characteristics; it is used only if
    ``survey_date`` is before the first day of the predicted month (same rule as training).
    """
    s = validate_history(history)
    target = s.index.max() + pd.offsets.MonthBegin(1)
    panel = pd.DataFrame({"household_ID": "INPUT", "month": list(s.index) + [target],
                          "consumption": list(s.values) + [np.nan]})
    df = add_calendar_features(add_history_features(panel))
    hf = pd.DataFrame([{"household_ID": "INPUT", **{c: (survey or {}).get(c, np.nan) for c in SURVEY_NUMERIC + SURVEY_CATEGORICAL}}])
    sd = pd.DataFrame({"household_ID": ["INPUT"],
                       "wave1": [pd.Timestamp(survey_date) if (survey and survey_date is not None) else pd.NaT]})
    df = add_survey_features(df, hf, sd)
    return df[df["month"] == target].reset_index(drop=True), target


def predict_next_month(bundle: dict, history: dict, survey: dict | None = None, survey_date=None) -> dict:
    row, target = build_feature_row(history, survey, survey_date)
    cols = model_input_columns(bundle["feature_set"])
    pred = float(bundle["pipeline"].predict(row[cols])[0])
    pred = max(pred, 0.0)
    return {"month": target, "prediction_kwh": pred, "features": row[cols].iloc[0].to_dict(),
            "n_bills_used": int(pd.Series(history).notna().sum())}
