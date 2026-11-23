"""Tests for the prediction path used by the dashboard (src/predict.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_survey_dates  # noqa: E402
from src.feature_engineering import HISTORY_FEATURES, CALENDAR_FEATURES, build_feature_table  # noqa: E402
from src.predict import InputError, build_feature_row, load_bundle, predict_next_month, validate_history  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
BUNDLE = ROOT / "models" / "best_model.joblib"
needs_data = pytest.mark.skipif(not (PROCESSED / "monthly_clean.parquet").exists(), reason="processed data missing")
needs_model = pytest.mark.skipif(not BUNDLE.exists(), reason="trained model not saved yet")


# ------------------------------------------------------------------ input validation
def test_missing_bills_stay_missing_not_zero():
    s = validate_history({"2024-01-01": 100, "2024-02-01": None, "2024-04-01": 90})   # March not given at all
    assert len(s) == 4                       # gap filled in as a month
    assert np.isnan(s.iloc[1]) and np.isnan(s.iloc[2])
    assert s.iloc[0] == 100 and s.iloc[3] == 90


@pytest.mark.parametrize("bad", [{}, {"2024-01-01": None}, {"2024-01-01": -5}, {"2024-01-01": 50_000}])
def test_invalid_inputs_raise_friendly_error(bad):
    with pytest.raises(InputError):
        validate_history(bad)


def test_target_is_month_after_last_bill_and_lags_are_correct():
    hist = {f"2024-{m:02d}-01": v for m, v in zip(range(1, 7), [10, 20, 30, 40, 50, 60])}
    row, target = build_feature_row(hist)
    assert target == pd.Timestamp("2024-07-01")
    r = row.iloc[0]
    assert r.lag_1 == 60 and r.lag_2 == 50 and r.lag_3 == 40 and r.lag_6 == 10
    assert r.roll_mean_3 == 50 and r.hist_max == 60 and r.month_of_year == 7
    assert np.isnan(r.lag_12)               # not enough history -> missing, imputed by the pipeline
    assert r.survey_available == 0


# ------------------------------------------------------------------ training/serving consistency
@needs_data
def test_prediction_features_match_training_feature_table():
    """Features built by the dashboard path must equal the training feature table for a real household."""
    m = pd.read_parquet(PROCESSED / "monthly_clean.parquet")
    hf = pd.read_parquet(PROCESSED / "household_features_w1.parquet")
    F = build_feature_table(m, hf, load_survey_dates())
    cols = HISTORY_FEATURES + CALENDAR_FEATURES
    rng = np.random.default_rng(0)
    for hh in rng.choice(m.household_ID.unique(), 15, replace=False):
        target = pd.Timestamp("2024-08-01")
        hist = m[(m.household_ID == hh) & (m.month < target)].set_index("month").consumption
        row, t = build_feature_row({k: (None if np.isnan(v) else v) for k, v in hist.items()})
        expected = F[(F.household_ID == hh) & (F.month == target)][cols].iloc[0]
        assert t == target
        pd.testing.assert_series_equal(row[cols].iloc[0].astype(float), expected.astype(float), check_names=False)


# ------------------------------------------------------------------ saved model
@needs_model
def test_saved_model_loads_and_predicts_reasonably():
    bundle = load_bundle(BUNDLE)
    meta = bundle["metadata"]
    assert meta["unit"] == "kWh" and meta["feature_columns"]
    stable = {f"2024-{m:02d}-01": 100.0 for m in range(1, 11)}
    out = predict_next_month(bundle, stable)
    assert out["month"] == pd.Timestamp("2024-11-01")
    assert 60 <= out["prediction_kwh"] <= 140          # a stable 100 kWh household should be predicted near 100
    gappy = dict(stable); gappy["2024-05-01"] = None; gappy["2024-07-01"] = None
    assert np.isfinite(predict_next_month(bundle, gappy)["prediction_kwh"])
    big = {f"2024-{m:02d}-01": 2000.0 for m in range(1, 11)}
    assert predict_next_month(bundle, big)["prediction_kwh"] > predict_next_month(bundle, stable)["prediction_kwh"]
    assert predict_next_month(bundle, {"2024-10-01": 0.0})["prediction_kwh"] >= 0   # unusual: single zero bill
