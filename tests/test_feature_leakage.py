"""Data-leakage checks for features, survey availability and splits.

These tests encode the leakage rules:
* a feature row for billing month t uses only bills of months <= t-1;
* Wave 1 survey features are used only if the interview date < first day of month t;
* train (<= 2024-05) and test (2024-06..2024-10) are separated in time;
* tuning folds validate on a month strictly after all their training months;
* identifiers, the target and data-quality flags are never model inputs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_survey_dates  # noqa: E402
from src.feature_engineering import (CALENDAR_FEATURES, FEATURE_SETS, FORBIDDEN_FEATURES,  # noqa: E402
                                     HISTORY_FEATURES, SURVEY_CATEGORICAL, SURVEY_NUMERIC,
                                     add_history_features, add_survey_features, build_feature_table)
from src.validation import (TEST_START, TRAIN_END, expanding_month_folds, modelling_rows,  # noqa: E402
                            train_test_split_time, unseen_household_folds)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
needs_data = pytest.mark.skipif(not (PROCESSED / "monthly_clean.parquet").exists(), reason="processed data missing")


def _synthetic_panel(n_households=6, n_months=18, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", periods=n_months, freq="MS")
    rows = []
    for h in range(n_households):
        vals = rng.integers(20, 300, n_months).astype(float)
        vals[rng.random(n_months) < 0.1] = np.nan      # some missing readings
        vals[rng.random(n_months) < 0.05] = 0           # some zeros
        rows += [(f"ID{h:04d}", mo, v) for mo, v in zip(months, vals)]
    return pd.DataFrame(rows, columns=["household_ID", "month", "consumption"])


# ------------------------------------------------------------------ history features
def test_lags_equal_actual_past_values():
    p = _synthetic_panel()
    f = add_history_features(p).set_index(["household_ID", "month"])
    actual = p.set_index(["household_ID", "month"])["consumption"]
    for (h, mo), row in f.sample(40, random_state=1).iterrows():
        for k in [1, 2, 3, 6, 12]:
            prev = actual.get((h, mo - pd.DateOffset(months=k)), np.nan)
            assert (np.isnan(prev) and np.isnan(row[f"lag_{k}"])) or prev == row[f"lag_{k}"]


@pytest.mark.parametrize("cut", [3, 7, 12, 17])
def test_features_do_not_change_when_current_or_future_values_change(cut):
    """Perturb consumption of months >= cut; every feature of months <= cut must be unchanged.

    Month == cut is included: the target month's own value must not influence its features.
    """
    p = _synthetic_panel()
    months = sorted(p.month.unique())
    boundary = months[cut]
    q = p.copy()
    future = q.month >= boundary
    q.loc[future, "consumption"] = q.loc[future, "consumption"] * 7 + 1000   # drastic change
    a = add_history_features(p).set_index(["household_ID", "month"])[HISTORY_FEATURES]
    b = add_history_features(q).set_index(["household_ID", "month"])[HISTORY_FEATURES]
    early = a.index.get_level_values("month") <= boundary
    pd.testing.assert_frame_equal(a[early], b[early])
    # sanity: the perturbation does change later features, so the test is not trivially passing
    later = a.index.get_level_values("month") > boundary
    if later.any():
        assert not a[later].equals(b[later])


def test_rolling_features_exclude_current_month():
    p = pd.DataFrame({"household_ID": "ID0001", "month": pd.date_range("2023-01-01", periods=5, freq="MS"),
                      "consumption": [10.0, 20.0, 30.0, 1000.0, 50.0]})
    f = add_history_features(p)
    row = f.iloc[3]                       # target month value is 1000
    assert row["roll_mean_3"] == 20.0     # mean(10, 20, 30) -- 1000 not included
    assert row["hist_max"] == 30.0
    assert row["lag_1"] == 30.0 and row["change_1"] == 10.0


# ------------------------------------------------------------------ survey availability
def test_survey_masked_before_interview_date():
    panel = pd.DataFrame({"household_ID": ["A"] * 4 + ["B"] * 4,
                          "month": list(pd.date_range("2024-01-01", periods=4, freq="MS")) * 2,
                          "consumption": 100.0})
    hf = pd.DataFrame({"household_ID": ["A", "B"]})
    for c in SURVEY_NUMERIC:
        hf[c] = 1.0
    for c in SURVEY_CATEGORICAL:
        hf[c] = "x"
    # A interviewed 2024-01-08 -> usable from Feb 2024; B interviewed exactly 2024-03-01 -> usable from Apr
    sd = pd.DataFrame({"household_ID": ["A", "B"], "wave1": pd.to_datetime(["2024-01-08", "2024-03-01"])})
    out = add_survey_features(panel, hf, sd).set_index(["household_ID", "month"])
    avail = out["survey_available"]
    assert list(avail.loc["A"]) == [0, 1, 1, 1]
    assert list(avail.loc["B"]) == [0, 0, 0, 1]     # same-day interview is NOT before the prediction date
    assert out.loc[avail == 0, SURVEY_NUMERIC + SURVEY_CATEGORICAL].isna().all().all()
    assert out.loc[avail == 1, SURVEY_NUMERIC].notna().all().all()


@needs_data
def test_real_feature_table_survey_rule_and_no_wave2_wave3():
    m = pd.read_parquet(PROCESSED / "monthly_clean.parquet")
    hf = pd.read_parquet(PROCESSED / "household_features_w1.parquet")
    F = build_feature_table(m, hf, load_survey_dates())
    on = F["survey_available"] == 1
    assert (F.loc[on, "wave1_date"] < F.loc[on, "month"]).all()
    assert F.loc[~on, SURVEY_NUMERIC].isna().all().all()
    assert not any(c.startswith(("w2_", "w3_", "wave2", "wave3")) for c in F.columns)
    assert not F.duplicated(["household_ID", "month"]).any()


# ------------------------------------------------------------------ feature sets & splits
def test_feature_sets_contain_no_forbidden_columns():
    for spec in FEATURE_SETS.values():
        cols = set(spec["numeric"]) | set(spec["categorical"])
        assert not cols & FORBIDDEN_FEATURES
    assert set(FEATURE_SETS["history_only"]["numeric"]) == set(HISTORY_FEATURES + CALENDAR_FEATURES)


@needs_data
def test_train_test_separated_in_time_and_folds_respect_order():
    m = pd.read_parquet(PROCESSED / "monthly_clean.parquet")
    hf = pd.read_parquet(PROCESSED / "household_features_w1.parquet")
    rows = modelling_rows(build_feature_table(m, hf, load_survey_dates()))
    train, test = train_test_split_time(rows)
    assert train.month.max() == TRAIN_END and test.month.min() == TEST_START
    assert train.month.max() < test.month.min()
    assert train.consumption.notna().all() and test.consumption.notna().all()
    n_folds = 0
    for tr_idx, va_idx in expanding_month_folds(train):
        n_folds += 1
        assert train.month.iloc[tr_idx].max() < train.month.iloc[va_idx].min()
        assert train.month.iloc[va_idx].nunique() == 1
    assert n_folds == 5
    for tr_mask, te_mask in unseen_household_folds(train, test):
        assert not set(train.household_ID[tr_mask]) & set(test.household_ID[te_mask])
