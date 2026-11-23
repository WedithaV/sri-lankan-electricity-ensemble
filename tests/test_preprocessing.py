"""Tests for the sklearn preprocessing pipeline: fit-on-train-only, unseen categories, missing values."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import CALENDAR_FEATURES, HISTORY_FEATURES, SURVEY_CATEGORICAL, SURVEY_NUMERIC  # noqa: E402
from src.preprocessing import build_preprocessor, model_input_columns  # noqa: E402


def _frame(n, seed, lag_value=None, category="SEC A"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(10, 300, size=(n, len(HISTORY_FEATURES + CALENDAR_FEATURES))),
                      columns=HISTORY_FEATURES + CALENDAR_FEATURES)
    if lag_value is not None:
        df["lag_1"] = lag_value
    df["survey_available"] = 1
    for c in SURVEY_NUMERIC:
        df[c] = rng.integers(0, 5, n).astype(float)
    for c in SURVEY_CATEGORICAL:
        df[c] = category
    return df


def test_imputer_statistics_come_from_training_rows_only():
    train = _frame(200, 0, lag_value=100.0)
    train.loc[:9, "lag_1"] = np.nan
    other = _frame(200, 1, lag_value=5000.0)       # very different data, e.g. the test period
    pre = build_preprocessor("history_only").fit(train[model_input_columns("history_only")])
    imputer = pre.named_transformers_["history"].named_steps["impute"]
    lag1_pos = list(pre.transformers_[0][2]).index("lag_1")
    assert imputer.statistics_[lag1_pos] == 100.0          # training median, not influenced by 'other'
    pre.transform(other[model_input_columns("history_only")])  # transforming must not refit
    assert imputer.statistics_[lag1_pos] == 100.0


def test_missing_values_imputed_with_indicator_and_no_nan_output():
    # sklearn adds an indicator only for columns that had missing values in TRAINING data
    # (as lag features do in the real training period); lag_2 has none here.
    train = _frame(100, 2)
    train.loc[:4, ["lag_1", "lag_12"]] = np.nan
    test = _frame(5, 3)
    test.loc[0, ["lag_1", "lag_2", "lag_12"]] = np.nan
    cols = model_input_columns("history_only")
    pre = build_preprocessor("history_only", scale=True).fit(train[cols])
    X = pre.transform(test[cols])
    assert not np.isnan(X).any()                      # lag_2 is still imputed even without an indicator
    names = list(pre.get_feature_names_out())
    assert any(n.endswith("missingindicator_lag_1") for n in names)
    assert not any(n.endswith("missingindicator_lag_2") for n in names)


def test_unseen_category_does_not_crash_and_maps_to_zeros():
    cols = model_input_columns("history_plus_survey")
    pre = build_preprocessor("history_plus_survey").fit(_frame(100, 4, category="SEC A")[cols])
    X = pre.transform(_frame(3, 5, category="A CATEGORY NEVER SEEN")[cols])
    names = list(pre.get_feature_names_out())
    sec = [i for i, n in enumerate(names) if n.startswith("survey_cat__socio_economic_class")]
    assert not np.isnan(X).any()
    assert X[:, sec].sum() == 0      # unknown category -> all-zero one-hot block
