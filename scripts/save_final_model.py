"""Save the selected model and all evaluation artifacts used by the dashboard.

Outputs
  models/best_model.joblib            final pipeline + metadata (compressed)
  models/ridge_baseline.joblib        small comparison model for the prediction page
  artifacts/model_results.csv         every model x experiment: CV and test metrics
  artifacts/cross_validation_results.csv   per-fold validation results
  artifacts/test_predictions.csv      test-period predictions of the final model and comparison models
  artifacts/experiment_results.csv    ablation / unseen households / robustness summaries
  artifacts/feature_importance.csv    permutation importance (test period) + tree importances
  artifacts/model_metadata.json       everything needed to describe/reload the final model

The final model is refitted exactly as in the evaluation and its test predictions are
compared with the saved evaluation predictions (must match) before anything is written.

Run from the project root:  python scripts/save_final_model.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiments import RIDGE_ALPHA, SELECTION, TUNING, build  # noqa: E402
from src.data_loader import CONSUMPTION_UNIT  # noqa: E402
from src.ensembles import TimeOrderedStackingRegressor  # noqa: E402
from src.evaluate import regression_metrics, rolling_test_predict  # noqa: E402
from src.feature_engineering import FEATURE_DESCRIPTIONS, FEATURE_SETS  # noqa: E402
from src.preprocessing import model_input_columns  # noqa: E402
from src.train import EXPERIMENTS, load_modelling_data, load_summaries  # noqa: E402
from src.validation import CV_VALIDATION_MONTHS, TEST_END, TEST_START, TRAIN_END, FIRST_TARGET_MONTH  # noqa: E402

ART, MODELS = ROOT / "artifacts", ROOT / "models"


def fit_final(label, feature_set, train):
    key, est, is_stack = build(label, feature_set)
    cols = model_input_columns(feature_set)
    if is_stack:
        est.fit(train[cols], train.consumption.to_numpy(), train.month.to_numpy())
        weights = est.meta_weights_.to_dict()
        # fold-specific base models and OOF rows are only needed for validation -> drop to keep the file small
        for attr in ("fold_bases_", "oof_"):
            if hasattr(est, attr):
                delattr(est, attr)
        return key, est, weights
    est.fit(train[cols], train.consumption.to_numpy())
    return key, est, None


def main():
    _, train, test = load_modelling_data()
    sel = json.loads(SELECTION.read_text())
    label, fs = sel["final_model"], "history_only"
    cols = model_input_columns(fs)

    t0 = time.perf_counter()
    key, model, weights = fit_final(label, fs, train)
    fit_s = time.perf_counter() - t0
    preds = rolling_test_predict(model, test, fs)
    saved = pd.read_parquet(EXPERIMENTS / f"testpred__tuned__{label}__{fs}.parquet")
    diff = np.abs(preds.predicted.to_numpy() - saved.predicted.to_numpy()).max()
    print(f"Refit check: max |difference| vs evaluated predictions = {diff:.2e} kWh")
    if diff > 1e-6:
        raise SystemExit("Refitted model does not reproduce the evaluated predictions - not saving.")
    m = regression_metrics(preds.actual, preds.predicted)

    summ = load_summaries("tuned").set_index("label").loc[label]
    bp = json.loads(TUNING.read_text())
    meta = {
        "model_label": label, "model_name": summ["model"], "family": summ["family"], "feature_set": fs,
        "feature_columns": cols, "feature_descriptions": {c: FEATURE_DESCRIPTIONS.get(c, "") for c in cols},
        "target": "consumption", "unit": CONSUMPTION_UNIT,
        "prediction_task": "one-month-ahead bill of a known household (billing month t, using bills up to t-1)",
        "selection_rule": sel["rule"],
        "training_period": [f"{FIRST_TARGET_MONTH:%Y-%m}", f"{TRAIN_END:%Y-%m}"],
        "validation_months": [f"{v:%Y-%m}" for v in CV_VALIDATION_MONTHS],
        "test_period": [f"{TEST_START:%Y-%m}", f"{TEST_END:%Y-%m}"],
        "training_rows": int(len(train)), "test_rows": int(len(test)),
        "cv_MAE": float(summ["cv_MAE"]), "cv_RMSE": float(summ["cv_RMSE"]), "cv_R2": float(summ["cv_R2"]),
        "test_MAE": m["MAE"], "test_RMSE": m["RMSE"], "test_R2": m["R2"],
        "hyperparameters": {"ridge_alpha": RIDGE_ALPHA, **bp}, "model_selection": sel,
        "stacking_meta_weights": weights,
        "dataset": {"name": "RECON-SL (Residential Electricity Consumption Dataset for Sri Lanka)",
                    "publisher": "LIRNEasia with Lanka Electricity Company (LECO)",
                    "doi": "10.21227/n1dk-q860", "zenodo_record": "15023048", "licence": "CC-BY-4.0",
                    "scope": "Households in LECO service areas along Sri Lanka's western/south-western coast"},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_seconds": fit_s,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "feature_set": fs, "metadata": meta}, MODELS / "best_model.joblib", compress=3)
    _, ridge, _ = fit_final("ridge", fs, train)
    joblib.dump({"pipeline": ridge, "feature_set": fs, "metadata": {"model_label": "ridge", "model_name": "Ridge Regression",
                 "unit": CONSUMPTION_UNIT, "hyperparameters": {"alpha": RIDGE_ALPHA}}}, MODELS / "ridge_baseline.joblib", compress=3)
    print("model sizes (MB):", {p.name: round(p.stat().st_size / 1e6, 1) for p in MODELS.glob("*.joblib")})

    # ---- results tables
    allsum = load_summaries()
    allsum.to_csv(ART / "model_results.csv", index=False)
    main_exps = allsum[allsum.experiment.isin(["main", "untuned", "tuned"])]
    main_exps.to_csv(ART / "model_results_main.csv", index=False)
    allsum[~allsum.experiment.isin(["main", "untuned", "tuned"])].to_csv(ART / "experiment_results.csv", index=False)
    cvs = pd.concat([pd.read_csv(p) for p in sorted(EXPERIMENTS.glob("cv__*.csv"))], ignore_index=True)
    cvs.to_csv(ART / "cross_validation_results.csv", index=False)

    # ---- test predictions (final + comparison models), wide format
    wide = preds.rename(columns={"predicted": f"pred_{label}"})
    comp = {"naive_last_month": "main", "ridge": "main"}
    for l in ["random_forest", "extra_trees", "gradient_boosting", "xgboost", "voting", "stacking"]:
        comp[l] = "tuned"
    for l, e in comp.items():
        if l == label:
            continue
        p = pd.read_parquet(EXPERIMENTS / f"testpred__{e}__{l}__{fs}.parquet")
        wide[f"pred_{l}"] = p.predicted.to_numpy()
    wide.to_csv(ART / "test_predictions.csv", index=False)

    # ---- feature importance
    t0 = time.perf_counter()
    pi = permutation_importance(model, test[cols], test.consumption.to_numpy(), scoring="neg_mean_absolute_error",
                                n_repeats=5, random_state=42, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "importance_std": pi.importances_std,
                        "method": "permutation (test period, increase in MAE kWh)", "model": label})
    print(f"permutation importance: {time.perf_counter() - t0:.0f}s")
    tree_rows = []
    # Built-in importances of every tree member (stacking bases / voting members / single model)
    pairs = []      # (member name, fitted estimator, feature names)
    if isinstance(model, TimeOrderedStackingRegressor):
        for name, pipe in model.bases_.items():
            if hasattr(pipe, "named_steps"):
                pairs.append((name, pipe.named_steps["model"], pipe.named_steps["preprocess"].get_feature_names_out()))
    else:
        names = model.named_steps["preprocess"].get_feature_names_out()
        est = model.named_steps["model"]
        if hasattr(est, "named_estimators_"):          # VotingRegressor
            pairs += [(n, e, names) for n, e in est.named_estimators_.items()]
        else:
            pairs.append((label, est, names))
    for name, est, names in pairs:
        if hasattr(est, "feature_importances_"):
            tree_rows.append(pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names],
                                           "importance": est.feature_importances_, "importance_std": np.nan,
                                           "method": "built-in (impurity / gain)", "model": name}))
    pd.concat([imp] + tree_rows, ignore_index=True).to_csv(ART / "feature_importance.csv", index=False)
    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ["model_label", "cv_MAE", "test_MAE", "test_RMSE", "test_R2"]}, indent=2))


if __name__ == "__main__":
    main()
