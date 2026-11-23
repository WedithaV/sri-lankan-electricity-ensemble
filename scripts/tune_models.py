"""Hyper-parameter tuning with RandomizedSearchCV on time-ordered folds.

* Folds: the 5 expanding-window one-month-ahead folds inside the training period
  (validate on Jan, Feb, Mar, Apr, May 2024; train only on earlier months).
  The test period (Jun-Oct 2024) is never seen.
* Selection metric: mean validation MAE (kWh). RMSE is recorded too.
* Besides tree settings, the search includes the LOSS FUNCTION (squared / absolute / Huber)
  and whether to train on a log target, because the untuned models optimised squared error
  while the project selects on MAE.
* Budget: RandomizedSearchCV samples a limited number of combinations (laptop-friendly);
  ranges are explained in SEARCH_SPACES below.

Run from the project root:  python scripts/tune_models.py [model ...]
Writes artifacts/tuning/<model>_search.csv and artifacts/tuning/best_params.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import SEED, make_tunable_pipeline  # noqa: E402
from src.preprocessing import model_input_columns  # noqa: E402
from src.train import load_modelling_data  # noqa: E402
from src.validation import expanding_month_folds  # noqa: E402

OUT = ROOT / "artifacts" / "tuning"
R = "model__regressor__"

# Ranges and reasons (also reproduced in notebook 06):
SEARCH_SPACES = {
    "random_forest": dict(n_iter=12, n_jobs=1, space={
        R + "n_estimators": [150, 300],              # more trees = more stable average; >300 adds memory, little gain
        R + "min_samples_leaf": [5, 10, 20, 40],     # bigger leaves = smoother, smaller model; untuned used 5
        R + "max_features": [0.33, 0.5, 0.8, 1.0],   # fewer features per split = more diverse (less correlated) trees
        R + "max_depth": [None, 12, 20],
        R + "max_samples": [0.5, 0.8, None],         # bootstrap sample size; smaller = faster, more diverse
        "model__log": [False, True],
    }),
    "extra_trees": dict(n_iter=12, n_jobs=1, space={
        R + "n_estimators": [150, 300],
        R + "min_samples_leaf": [5, 10, 20, 40],
        R + "max_features": [0.33, 0.5, 0.8, 1.0],
        R + "max_depth": [None, 12, 20],
        "model__log": [False, True],
    }),
    "gradient_boosting": dict(n_iter=16, n_jobs=6, space={
        R + "loss": ["squared_error", "absolute_error", "huber"],  # absolute error optimises MAE directly
        R + "n_estimators": [100, 200, 400, 600],
        R + "learning_rate": loguniform(0.02, 0.2),  # smaller rate needs more trees but generalises better
        R + "max_depth": randint(2, 6),               # 2-5: boosting uses shallow "weak" trees
        R + "subsample": uniform(0.6, 0.4),           # 0.6-1.0: stochastic boosting reduces over-fitting
        R + "min_samples_leaf": [1, 10, 30, 100],
        "model__log": [False, True],
    }),
    "xgboost": dict(n_iter=30, n_jobs=1, space={
        R + "objective": ["reg:squarederror", "reg:absoluteerror"],
        R + "n_estimators": [200, 400, 800, 1200],
        R + "learning_rate": loguniform(0.01, 0.2),
        R + "max_depth": randint(3, 9),               # untuned default 6 over-fitted
        R + "subsample": uniform(0.6, 0.4),
        R + "colsample_bytree": uniform(0.5, 0.5),    # 0.5-1.0 of features per tree
        R + "min_child_weight": [1, 5, 10, 20, 50],   # minimum (hessian) weight per leaf -> larger = more conservative
        R + "reg_lambda": loguniform(0.1, 50),        # L2 penalty on leaf values
        "model__log": [False, True],
    }),
}


def tune(name: str, train: pd.DataFrame, feature_set: str = "history_only") -> dict:
    cfg = SEARCH_SPACES[name]
    cols = model_input_columns(feature_set)
    folds = list(expanding_month_folds(train))
    search = RandomizedSearchCV(
        make_tunable_pipeline(name, feature_set), cfg["space"], n_iter=cfg["n_iter"], cv=folds,
        scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"},
        refit=False, random_state=SEED, n_jobs=cfg["n_jobs"], return_train_score=False, verbose=0)
    t0 = time.perf_counter()
    search.fit(train[cols], train["consumption"].to_numpy())
    secs = time.perf_counter() - t0
    res = pd.DataFrame(search.cv_results_)
    res["cv_MAE"] = -res["mean_test_MAE"]
    res["cv_RMSE"] = -res["mean_test_RMSE"]
    keep = [c for c in res.columns if c.startswith("param_")] + ["cv_MAE", "std_test_MAE", "cv_RMSE", "mean_fit_time"]
    res = res[keep].sort_values("cv_MAE")
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"{name}_search.csv", index=False)
    best = res.iloc[0]
    params = {c.replace("param_" + R, ""): (v.item() if hasattr(v, "item") else v)
              for c, v in best.items() if c.startswith("param_" + R)}
    params = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in params.items()}
    out = {"log": bool(best["param_model__log"]), "params": params, "cv_MAE": float(best["cv_MAE"]),
           "cv_RMSE": float(best["cv_RMSE"]), "n_candidates": int(len(res)), "search_seconds": secs}
    print(f"{name}: best cv_MAE {out['cv_MAE']:.3f} (RMSE {out['cv_RMSE']:.2f}) in {secs/60:.1f} min -> {out}", flush=True)
    return out


def main(names):
    _, train, _ = load_modelling_data()
    path = OUT / "best_params.json"
    best = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        best[n] = tune(n, train)
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or ["xgboost", "gradient_boosting", "extra_trees", "random_forest"])
