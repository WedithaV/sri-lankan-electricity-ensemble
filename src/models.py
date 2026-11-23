"""Model definitions: naive baselines, linear baselines and ensemble regressors.

Every model is returned as an sklearn Pipeline  (preprocessor -> estimator), so that
preprocessing is fitted on training rows only and the same object can be saved and used
for prediction in the dashboard.

Naive baselines work directly on the engineered feature columns (no learning except a
fallback median), so they need no preprocessor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor

SEED = 42
N_JOBS = -1

# Display names and family (used for colours/grouping in reports and the dashboard)
MODEL_INFO = {
    "naive_last_month":   ("Naive: last month's bill", "Baseline"),
    "naive_mean_3m":      ("Naive: mean of last 3 months", "Baseline"),
    "dummy_mean":         ("DummyRegressor (training mean)", "Baseline"),
    "linear":             ("Linear Regression", "Baseline"),
    "ridge":              ("Ridge Regression", "Baseline"),
    "random_forest":      ("Random Forest", "Bagging"),
    "extra_trees":        ("Extra Trees", "Bagging"),
    "gradient_boosting":  ("Gradient Boosting", "Boosting"),
    "xgboost":            ("XGBoost", "Boosting"),
    "hist_gradient_boosting": ("Histogram Gradient Boosting", "Boosting"),
    "voting":             ("Voting Regressor", "Ensemble combination"),
    "stacking":           ("Stacking (time-ordered, Ridge meta-learner)", "Ensemble combination"),
}


# ---------------------------------------------------------------- naive baselines
class NaiveHistoryRegressor(RegressorMixin, BaseEstimator):
    """Predict with an existing history column, falling back when it is missing.

    column="lag_1"       -> "this month = last month's bill"   (persistence forecast)
    column="roll_mean_3" -> "this month = mean of the last 3 bills"
    Fallback order: the column, then roll_mean_3, then hist_mean, then the training median.
    """

    def __init__(self, column: str = "lag_1"):
        self.column = column

    def fit(self, X: pd.DataFrame, y):
        self.fallback_ = float(np.median(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame):
        pred = X[self.column].copy()
        for col in ["roll_mean_3", "hist_mean"]:
            if col in X:
                pred = pred.fillna(X[col])
        return pred.fillna(self.fallback_).to_numpy(dtype=float)


# ---------------------------------------------------------------- optional log target
class OptionalLogTarget(RegressorMixin, BaseEstimator):
    """Wrap a regressor; if ``log=True`` it is trained on log(1 + kWh) and predictions are
    converted back to kWh with exp(x) - 1. Making this a parameter lets the tuning search
    decide whether the log target helps. Predictions are clipped at 0 kWh (a bill cannot be
    negative)."""

    def __init__(self, regressor=None, log: bool = False):
        self.regressor = regressor
        self.log = log

    def fit(self, X, y):
        from sklearn.base import clone
        y = np.asarray(y, float)
        self.regressor_ = clone(self.regressor).fit(X, np.log1p(np.clip(y, 0, None)) if self.log else y)
        return self

    def predict(self, X):
        p = self.regressor_.predict(X)
        return np.clip(np.expm1(p) if self.log else p, 0, None)

    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_


# ---------------------------------------------------------------- default (untuned) estimators
def default_estimator(name: str):
    """Library defaults, plus a fixed seed / parallelism. These are the UNTUNED versions."""
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    # Forests: sklearn's defaults grow every leaf down to one row. Measured on this data that
    # needs ~1.7 GB (Random Forest) / ~2.8 GB (Extra Trees) for 300 trees -- not feasible on a
    # standard laptop or for deployment. Starting configuration: 200 trees, >= 5 rows per leaf
    # (~200 MB). This is a memory constraint, not tuning; scripts/tune_models.py tunes these settings.
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=N_JOBS)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=N_JOBS)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=SEED)
    if name == "xgboost":
        return XGBRegressor(random_state=SEED, n_jobs=N_JOBS, tree_method="hist")
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=SEED)
    raise ValueError(f"Unknown model: {name}")


LINEAR_MODELS = {"linear", "ridge", "dummy_mean"}


def make_pipeline(name: str, feature_set: str = "history_only", params: dict | None = None,
                  log_target: bool = False):
    """Build preprocessor + estimator. ``params`` are estimator hyper-parameters (e.g. from tuning)."""
    if name == "naive_last_month":
        return NaiveHistoryRegressor("lag_1")
    if name == "naive_mean_3m":
        return NaiveHistoryRegressor("roll_mean_3")
    est = default_estimator(name)
    if params:
        est.set_params(**params)
    if log_target:
        # Fit on log(1 + kWh), predict back in kWh. Reduces the influence of very large households.
        est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
    pre = build_preprocessor(feature_set, scale=name in LINEAR_MODELS)
    return Pipeline([("preprocess", pre), ("model", est)])


def make_tunable_pipeline(name: str, feature_set: str = "history_only"):
    """Pipeline used by the tuning search: preprocess -> OptionalLogTarget(estimator).

    Search parameters are named ``model__log`` and ``model__regressor__<hyper-parameter>``.
    """
    pre = build_preprocessor(feature_set, scale=name in LINEAR_MODELS)
    return Pipeline([("preprocess", pre), ("model", OptionalLogTarget(default_estimator(name)))])


def make_tuned_pipeline(name: str, tuned: dict, feature_set: str = "history_only"):
    """Rebuild a tuned model from saved best parameters {"log": bool, "params": {...}}."""
    est = default_estimator(name).set_params(**tuned["params"])
    pre = build_preprocessor(feature_set, scale=name in LINEAR_MODELS)
    return Pipeline([("preprocess", pre), ("model", OptionalLogTarget(est, log=tuned["log"]))])


def make_voting(members: dict[str, dict], feature_set: str = "history_only"):
    """VotingRegressor averaging the predictions of already-chosen (tuned) tree ensembles.

    ``members`` maps model name -> hyper-parameters. Each member is a full estimator; the
    shared preprocessor is applied once before voting.
    """
    estimators = []
    for name, cfg in members.items():
        if cfg and "params" in cfg:          # tuned member: {"log": bool, "params": {...}}
            est = OptionalLogTarget(default_estimator(name).set_params(**cfg["params"]), log=cfg["log"])
        else:
            est = default_estimator(name)
            if cfg:
                est.set_params(**cfg)
        estimators.append((name, est))
    pre = build_preprocessor(feature_set, scale=False)
    return Pipeline([("preprocess", pre), ("model", VotingRegressor(estimators, n_jobs=1))])
