"""Time-ordered stacking.

WHY NOT sklearn's StackingRegressor?
sklearn builds the meta-learner's training data with ordinary K-fold cross_val_predict:
every row is predicted by base models trained on all OTHER folds -- including LATER
months. For a forecasting problem that is temporal leakage (a base model would learn from
Mar 2024 bills to produce the "out-of-fold" prediction for Jan 2024). It also requires
folds that partition all rows, which expanding-window folds do not.

HOW THIS VERSION WORKS
1. Base models are diverse learners (e.g. naive persistence, Ridge, a bagging forest and a
   boosting model).
2. Out-of-fold (OOF) predictions are produced month by month: for each month v in
   ``oof_months``, every base model is trained ONLY on months before v and predicts month
   v. So every OOF prediction is a genuine "future" prediction, exactly as at test time.
3. The meta-learner (Ridge, non-negative weights) learns how to combine the base
   predictions from these OOF rows: y ~ w1*pred1 + w2*pred2 + ... + b.
4. Finally each base model is refitted on the whole training period. At prediction time,
   the base models predict and the meta-learner combines their predictions.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge

from src.evaluate import regression_metrics, rolling_test_predict
from src.preprocessing import model_input_columns
from src.validation import CV_VALIDATION_MONTHS

# Starts in Nov 2023: the first month whose earlier training months include rows with lag_12
# (lag_12 exists from target month Oct 2023). For Oct 2023 the base models would see lag_12 as
# entirely empty and the imputer would drop it. 7 OOF months (~28,000 rows) train the meta-learner.
DEFAULT_OOF_MONTHS = pd.date_range("2023-11-01", "2024-05-01", freq="MS")


class TimeOrderedStackingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, base_models: dict, meta_model=None, oof_months=DEFAULT_OOF_MONTHS):
        self.base_models = base_models
        self.meta_model = meta_model
        self.oof_months = oof_months

    def _meta(self):
        return clone(self.meta_model) if self.meta_model is not None else Ridge(alpha=1.0, positive=True)

    def _base_oof(self, X: pd.DataFrame, y: np.ndarray, months: np.ndarray, oof_months) -> pd.DataFrame:
        """OOF base predictions for each month in oof_months (bases trained on earlier months only).

        Also keeps, per month, the base models fitted on data before that month (needed for
        time-ordered CV of the whole stack).
        """
        frames, self.fold_bases_ = [], {}
        for v in oof_months:
            v = np.datetime64(v)
            tr, va = months < v, months == v
            if not va.any():
                continue
            preds, fitted = {}, {}
            for name, model in self.base_models.items():
                m = clone(model).fit(X[tr], y[tr])
                preds[name] = m.predict(X[va])
                fitted[name] = m
            self.fold_bases_[pd.Timestamp(v)] = fitted
            frames.append(pd.DataFrame(preds).assign(month=months[va], y=y[va], row=np.flatnonzero(va)))
        return pd.concat(frames, ignore_index=True)

    def fit(self, X: pd.DataFrame, y, months):
        y = np.asarray(y, float)
        months = np.asarray(months, dtype="datetime64[ns]")
        names = list(self.base_models)
        self.oof_ = self._base_oof(X, y, months, self.oof_months)
        self.meta_ = self._meta().fit(self.oof_[names].to_numpy(), self.oof_["y"].to_numpy())
        self.bases_ = {n: clone(m).fit(X, y) for n, m in self.base_models.items()}
        return self

    def predict(self, X: pd.DataFrame):
        P = np.column_stack([self.bases_[n].predict(X) for n in self.base_models])
        return self.meta_.predict(P)

    @property
    def meta_weights_(self) -> pd.Series:
        return pd.Series(list(self.meta_.coef_) + [self.meta_.intercept_], index=list(self.base_models) + ["intercept"])


def evaluate_stacking(stack: TimeOrderedStackingRegressor, train: pd.DataFrame, test: pd.DataFrame,
                      feature_set: str, validation_months=CV_VALIDATION_MONTHS):
    """Time-ordered CV of the whole stack + full fit + rolling test (same outputs as train.run_model).

    For validation month v: base predictions for v come from base models trained on months < v;
    the meta-learner is trained on OOF rows of months < v only. Nothing from month v or later
    is used to build the prediction for v.
    """
    cols = model_input_columns(feature_set)
    X, y = train[cols], train["consumption"].to_numpy()
    months = train["month"].to_numpy(dtype="datetime64[ns]")
    names = list(stack.base_models)

    t0 = time.perf_counter()
    stack.fit(X, y, months)
    fit_s = time.perf_counter() - t0
    oof = stack.oof_

    rows, oof_out = [], []
    for i, v in enumerate(validation_months, start=1):
        v = pd.Timestamp(v)
        before, at = oof["month"] < v, oof["month"] == v
        meta_v = stack._meta().fit(oof.loc[before, names].to_numpy(), oof.loc[before, "y"].to_numpy())
        p_va = meta_v.predict(oof.loc[at, names].to_numpy())
        # training error of the fold's full system on a sample of its training rows
        bases_v = stack.fold_bases_[v]
        tr_idx = np.flatnonzero(months < np.datetime64(v))
        sample = np.random.default_rng(i).choice(tr_idx, size=min(len(tr_idx), 20000), replace=False)
        p_tr = meta_v.predict(np.column_stack([bases_v[n].predict(X.iloc[sample]) for n in names]))
        val, trn = regression_metrics(oof.loc[at, "y"], p_va), regression_metrics(y[sample], p_tr)
        rows.append({"fold": i, "validation_month": v.strftime("%Y-%m"), "train_rows": len(tr_idx),
                     "val_MAE": val["MAE"], "val_RMSE": val["RMSE"], "val_R2": val["R2"],
                     "train_MAE": trn["MAE"], "train_RMSE": trn["RMSE"], "train_R2": trn["R2"],
                     "fit_seconds": np.nan})
        r = oof.loc[at, "row"].to_numpy()
        oof_out.append(pd.DataFrame({"household_ID": train["household_ID"].iloc[r].to_numpy(),
                                     "month": train["month"].iloc[r].to_numpy(),
                                     "actual": y[r], "predicted": p_va}))
    cv = pd.DataFrame(rows)
    t0 = time.perf_counter()
    preds = rolling_test_predict(stack, test, feature_set)
    pred_s = time.perf_counter() - t0
    info = {"fit_seconds_full": fit_s, "predict_ms_per_1000_rows": 1000 * pred_s / len(test) * 1000}
    return stack, cv, pd.concat(oof_out, ignore_index=True), preds, info
