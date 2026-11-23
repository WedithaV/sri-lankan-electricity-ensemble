"""Experiment runner shared by the notebooks.

``run_model`` performs, for one model and one feature set:
  1. expanding-window CV on the training period (model-selection evidence),
  2. a fit on the full training period,
  3. rolling one-month-ahead prediction of the untouched test period.
Results are appended to artifacts/experiments/*.csv so that later notebooks and the
dashboard read saved, real results instead of re-training.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.data_loader import load_survey_dates
from src.evaluate import cross_validate_time, fit_and_test, regression_metrics
from src.feature_engineering import build_feature_table
from src.models import MODEL_INFO
from src.validation import modelling_rows, train_test_split_time

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
EXPERIMENTS = ARTIFACTS / "experiments"


@lru_cache(maxsize=1)
def load_modelling_data():
    """Feature table, training rows and test rows (cached for the session)."""
    monthly = pd.read_parquet(PROCESSED / "monthly_clean.parquet")
    hh = pd.read_parquet(PROCESSED / "household_features_w1.parquet")
    F = build_feature_table(monthly, hh, load_survey_dates())
    train, test = train_test_split_time(modelling_rows(F))
    return F, train.reset_index(drop=True), test.reset_index(drop=True)


def run_model(key: str, model, feature_set: str = "history_only", experiment: str = "main",
              train: pd.DataFrame | None = None, test: pd.DataFrame | None = None, save: bool = True,
              label: str | None = None):
    """Run CV + full fit + rolling test for one model. Returns (summary dict, cv table, test preds, fitted model)."""
    if train is None or test is None:
        _, train, test = load_modelling_data()
    cv, oof = cross_validate_time(model, train, feature_set)
    fitted, preds, info = fit_and_test(model, train, test, feature_set)
    summary = _summarise(key, label, experiment, feature_set, cv, preds, info)
    if save:
        save_experiment(summary, cv, oof, preds)
    return summary, cv, preds, fitted


def run_stacking(stack, feature_set: str = "history_only", experiment: str = "main", label: str = "stacking",
                 train: pd.DataFrame | None = None, test: pd.DataFrame | None = None, save: bool = True):
    """Same outputs as run_model, using the time-ordered stacking procedure (src/ensembles.py)."""
    from src.ensembles import evaluate_stacking
    if train is None or test is None:
        _, train, test = load_modelling_data()
    fitted, cv, oof, preds, info = evaluate_stacking(stack, train, test, feature_set)
    summary = _summarise("stacking", label, experiment, feature_set, cv, preds, info)
    if save:
        save_experiment(summary, cv, oof, preds)
    return summary, cv, preds, fitted


def _summarise(key, label, experiment, feature_set, cv, preds, info) -> dict:
    tm = regression_metrics(preds.actual, preds.predicted)
    name, family = MODEL_INFO.get(key, (key, "Other"))
    return {
        "experiment": experiment, "model_key": key, "label": label or key, "model": name, "family": family,
        "feature_set": feature_set,
        "cv_MAE": cv.val_MAE.mean(), "cv_MAE_std": cv.val_MAE.std(), "cv_RMSE": cv.val_RMSE.mean(),
        "cv_R2": cv.val_R2.mean(), "train_MAE": cv.train_MAE.mean(),
        "test_MAE": tm["MAE"], "test_RMSE": tm["RMSE"], "test_R2": tm["R2"], "test_n": tm["n"],
        "cv_fit_seconds_mean": cv.fit_seconds.mean(), **info,
    }


def save_experiment(summary: dict, cv: pd.DataFrame, oof: pd.DataFrame, preds: pd.DataFrame) -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    e, l, fs, k = summary["experiment"], summary["label"], summary["feature_set"], summary["model_key"]
    tag = f"{e}__{l}__{fs}"
    cv.assign(experiment=e, model_key=k, label=l, feature_set=fs).to_csv(EXPERIMENTS / f"cv__{tag}.csv", index=False)
    preds.assign(experiment=e, model_key=k, label=l, feature_set=fs).to_parquet(EXPERIMENTS / f"testpred__{tag}.parquet", index=False)
    oof.to_parquet(EXPERIMENTS / f"oof__{tag}.parquet", index=False)
    (EXPERIMENTS / f"summary__{tag}.json").write_text(json.dumps(summary, indent=2, default=str))


def load_summaries(experiment: str | None = None) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(EXPERIMENTS.glob("summary__*.json"))]
    df = pd.DataFrame(rows)
    if experiment is not None and len(df):
        df = df[df.experiment == experiment]
    return df.reset_index(drop=True)


def load_test_predictions(experiment: str, label: str, feature_set: str = "history_only") -> pd.DataFrame:
    return pd.read_parquet(EXPERIMENTS / f"testpred__{experiment}__{label}__{feature_set}.parquet")


def load_cv(experiment: str, label: str, feature_set: str = "history_only") -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS / f"cv__{experiment}__{label}__{feature_set}.csv")
