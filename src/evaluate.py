"""Evaluation helpers: metrics, time-ordered cross-validation and rolling one-month-ahead testing.

All errors are in kWh (the unit of ``consumption``).
* MAE  = average absolute error -- "on average the prediction is X kWh away from the bill".
* RMSE = square root of the average squared error -- punishes large misses more (sensitive
         to the few very large households).
* R^2  = share of the variance of the bills explained by the model (1 = perfect, 0 = no
         better than always predicting the mean of the evaluated data).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import model_input_columns
from src.validation import expanding_month_folds


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred), "n": int(len(y_true))}


def cross_validate_time(model, train: pd.DataFrame, feature_set: str, folds=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window one-month-ahead CV inside the training period.

    Returns (per-fold metrics, out-of-fold predictions for the validation months).
    Training error is measured on each fold's own training rows (for over-fitting checks).
    """
    cols = model_input_columns(feature_set)
    X, y = train[cols], train["consumption"].to_numpy()
    folds = list(folds if folds is not None else expanding_month_folds(train))
    rows, oof = [], []
    for i, (tr, va) in enumerate(folds, start=1):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y[tr])
        fit_s = time.perf_counter() - t0
        p_va = m.predict(X.iloc[va])
        # training error on a sample of the fold's training rows (cheap but representative)
        rng = np.random.default_rng(i)
        sample = rng.choice(tr, size=min(len(tr), 20000), replace=False)
        p_tr = m.predict(X.iloc[sample])
        val = regression_metrics(y[va], p_va)
        trn = regression_metrics(y[sample], p_tr)
        rows.append({"fold": i, "validation_month": train["month"].iloc[va[0]].strftime("%Y-%m"),
                     "train_rows": len(tr), "val_MAE": val["MAE"], "val_RMSE": val["RMSE"], "val_R2": val["R2"],
                     "train_MAE": trn["MAE"], "train_RMSE": trn["RMSE"], "train_R2": trn["R2"], "fit_seconds": fit_s})
        oof.append(pd.DataFrame({"household_ID": train["household_ID"].iloc[va].to_numpy(),
                                 "month": train["month"].iloc[va].to_numpy(),
                                 "actual": y[va], "predicted": p_va}))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def rolling_test_predict(fitted_model, test: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Rolling one-month-ahead predictions for the test period with ONE fixed fitted model.

    Months are processed in order. The feature row for month t was built only from bills up
    to t-1 (see feature_engineering), so month t's actual bill is used as an input only for
    later months, after month t has been predicted and evaluated.
    """
    cols = model_input_columns(feature_set)
    out = []
    for month in sorted(test["month"].unique()):
        rows = test[test["month"] == month]
        out.append(pd.DataFrame({"household_ID": rows["household_ID"].to_numpy(), "month": rows["month"].to_numpy(),
                                 "actual": rows["consumption"].to_numpy(),
                                 "predicted": fitted_model.predict(rows[cols])}))
    return pd.concat(out, ignore_index=True)


def fit_and_test(model, train: pd.DataFrame, test: pd.DataFrame, feature_set: str):
    """Fit on the full training period, then rolling-predict the test period. Returns (model, preds, info)."""
    cols = model_input_columns(feature_set)
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(train[cols], train["consumption"].to_numpy())
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = rolling_test_predict(m, test, feature_set)
    pred_s = time.perf_counter() - t0
    info = {"fit_seconds_full": fit_s, "predict_ms_per_1000_rows": 1000 * pred_s / len(test) * 1000}
    return m, preds, info


def per_month_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    return preds.groupby("month").apply(lambda d: pd.Series(regression_metrics(d.actual, d.predicted)),
                                        include_groups=False).reset_index()
