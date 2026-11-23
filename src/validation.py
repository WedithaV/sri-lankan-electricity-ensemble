"""Train/validation/test splitting.

* Target months start in January 2023, so every training row has at least 3 earlier
  billing months available for lag features (the panel starts in October 2022).
* TRAINING + VALIDATION: billing months 2023-01 .. 2024-05.
* TEST (untouched until final evaluation): billing months 2024-06 .. 2024-10, evaluated as
  rolling one-month-ahead predictions with one fixed model trained on the training period.
* Hyper-parameter tuning / model selection: expanding-window, one-month-ahead folds inside
  the training period (validate on each of 2024-01 .. 2024-05, training only on earlier
  months). The test months are never used.
* Rows whose target is missing are dropped (there is nothing to learn or evaluate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

FIRST_TARGET_MONTH = pd.Timestamp("2023-01-01")
TRAIN_END = pd.Timestamp("2024-05-01")
TEST_START = pd.Timestamp("2024-06-01")
TEST_END = pd.Timestamp("2024-10-01")
CV_VALIDATION_MONTHS = pd.date_range("2024-01-01", "2024-05-01", freq="MS")


def modelling_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as examples: target observed and month >= FIRST_TARGET_MONTH."""
    return features[(features["month"] >= FIRST_TARGET_MONTH) & features["consumption"].notna()].copy()


def train_test_split_time(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = rows[rows["month"] <= TRAIN_END].copy()
    test = rows[(rows["month"] >= TEST_START) & (rows["month"] <= TEST_END)].copy()
    return train, test


def expanding_month_folds(train: pd.DataFrame, validation_months=CV_VALIDATION_MONTHS):
    """Yield (train_idx, val_idx) positional indices: train on months < v, validate on month v.

    Usable as the ``cv`` argument of sklearn's RandomizedSearchCV / cross_val_score.
    """
    months = train["month"].to_numpy()
    for v in validation_months:
        v = np.datetime64(v)
        tr = np.flatnonzero(months < v)
        va = np.flatnonzero(months == v)
        if len(tr) and len(va):
            yield tr, va


def unseen_household_folds(train: pd.DataFrame, test: pd.DataFrame, n_splits: int = 5):
    """Scenario B: generalisation to households NOT used for training.

    Households are divided into ``n_splits`` groups. For each group g, a model is trained on
    TRAINING-period rows of the other households and evaluated on TEST-period rows of the
    households in g. Time order is still respected (train months < test months).
    Yields (train_rows_mask, test_rows_mask) as boolean arrays aligned with train/test.
    """
    households = np.array(sorted(set(train["household_ID"]) | set(test["household_ID"])))
    gkf = GroupKFold(n_splits=n_splits)
    for _, held_idx in gkf.split(households, groups=households):
        held = set(households[held_idx])
        yield ~train["household_ID"].isin(held).to_numpy(), test["household_ID"].isin(held).to_numpy()
