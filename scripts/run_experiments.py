"""Experiments with the tuned models.

Stages (run in order; each writes to artifacts/experiments/):
  tuned       tuned single ensembles + Voting + time-ordered Stacking built from tuned members
  ablation    history-only vs history + Wave 1 survey (date-masked) for key models
  unseen      Scenario B: households never seen in training (GroupKFold over households)
  robustness  R1: exclude the 34 always-zero/low-activity households; R2: drop isolated spikes from training targets

MODEL SELECTION RULE (fixed before looking at tuned test results): the final model is the
one with the LOWEST mean validation MAE (5 expanding folds) in the "tuned" experiment.
Test-set results are reported for every model but never used for choosing.

Run:  python scripts/run_experiments.py tuned ablation unseen robustness
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import flag_isolated_spikes  # noqa: E402
from src.ensembles import TimeOrderedStackingRegressor  # noqa: E402
from src.evaluate import regression_metrics, rolling_test_predict  # noqa: E402
from src.models import MODEL_INFO, make_pipeline, make_tuned_pipeline, make_voting  # noqa: E402
from src.preprocessing import model_input_columns  # noqa: E402
from src.train import EXPERIMENTS, load_modelling_data, load_summaries, run_model, run_stacking  # noqa: E402
from src.validation import TRAIN_END, unseen_household_folds  # noqa: E402

TUNING = ROOT / "artifacts" / "tuning" / "best_params.json"
SELECTION = ROOT / "artifacts" / "model_selection.json"
RIDGE_ALPHA = 1000.0          # selected in notebook 04 (validation folds)
SINGLE = ["random_forest", "extra_trees", "gradient_boosting", "xgboost"]
BAGGING, BOOSTING = ["random_forest", "extra_trees"], ["gradient_boosting", "xgboost"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_params():
    return json.loads(TUNING.read_text())


def build(label: str, feature_set: str = "history_only"):
    """Return (model_key, estimator, is_stacking) for any label used in the experiments."""
    bp = best_params()
    if label in ("naive_last_month", "naive_mean_3m", "dummy_mean", "linear"):
        return label, make_pipeline(label, feature_set), False
    if label == "ridge":
        return "ridge", make_pipeline("ridge", feature_set, params={"alpha": RIDGE_ALPHA}), False
    if label in SINGLE:
        return label, make_tuned_pipeline(label, bp[label], feature_set), False
    sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    if label == "voting":
        return "voting", make_voting({m: bp[m] for m in sel["voting_members"]}, feature_set), False
    if label == "stacking":
        bases = {"naive_last_month": make_pipeline("naive_last_month", feature_set),
                 "ridge": make_pipeline("ridge", feature_set, params={"alpha": RIDGE_ALPHA})}
        for m in sel["stacking_tree_bases"]:
            bases[m] = make_tuned_pipeline(m, bp[m], feature_set)
        return "stacking", TimeOrderedStackingRegressor(bases), True
    raise ValueError(label)


def run(label, experiment, feature_set="history_only", train=None, test=None):
    done = EXPERIMENTS / f"summary__{experiment}__{label}__{feature_set}.json"
    if done.exists():          # resume support: results are deterministic (fixed seeds)
        log(f"{experiment:12s} {label:18s} {feature_set:20s} already done - skipped")
        return json.loads(done.read_text())
    key, est, is_stack = build(label, feature_set)
    t0 = time.time()
    if is_stack:
        s, *_ = run_stacking(est, feature_set, experiment, label, train=train, test=test)
    else:
        s, *_ = run_model(key, est, feature_set, experiment, train=train, test=test, label=label)
    log(f"{experiment:12s} {label:18s} {feature_set:20s} cv_MAE {s['cv_MAE']:.3f}  test_MAE {s['test_MAE']:.3f}  "
        f"test_RMSE {s['test_RMSE']:.2f}  ({time.time()-t0:.0f}s)")
    return s


# ------------------------------------------------------------------ stages
def stage_tuned():
    for m in SINGLE:
        run(m, "tuned")
    s = load_summaries("tuned").set_index("label")
    singles = s.loc[SINGLE].sort_values("cv_MAE")
    # Voting: the three best tuned single ensembles (validation MAE).
    # Stacking tree bases: the best bagging and the best boosting model (diverse families),
    # plus naive persistence and Ridge as non-tree base learners.
    sel = {"voting_members": list(singles.index[:3]),
           "stacking_tree_bases": [singles.loc[BAGGING].cv_MAE.idxmin(), singles.loc[BOOSTING].cv_MAE.idxmin()]}
    SELECTION.write_text(json.dumps(sel, indent=2))
    log(f"members chosen from validation MAE: {sel}")
    run("voting", "tuned")
    run("stacking", "tuned")
    s = load_summaries("tuned")
    best = s.sort_values("cv_MAE").iloc[0]
    sel.update(final_model=best["label"], final_cv_MAE=float(best["cv_MAE"]),
               rule="lowest mean validation MAE over the 5 expanding folds (experiment 'tuned')")
    SELECTION.write_text(json.dumps(sel, indent=2))
    log(f"FINAL MODEL (by validation MAE): {best['label']} cv_MAE {best['cv_MAE']:.3f}")


def final_label():
    return json.loads(SELECTION.read_text())["final_model"]


def stage_ablation():
    labels = ["ridge", "random_forest", "extra_trees", "gradient_boosting", "xgboost"]
    fin = final_label()
    if fin not in labels:
        labels.append(fin)
    for lab in labels:
        run(lab, "ablation", "history_plus_survey")


def stage_unseen():
    """Scenario B. Each household is held out once; the model never sees its rows in training."""
    _, train, test = load_modelling_data()
    s = load_summaries("tuned").set_index("label")
    best_single = s.loc[SINGLE].cv_MAE.idxmin()
    fin = final_label()
    plan = [("naive_last_month", "history_only"), ("ridge", "history_only"), ("ridge", "history_plus_survey"),
            (best_single, "history_only"), (best_single, "history_plus_survey")]
    if fin != best_single:          # a combination model: retrained 5x, so history-only features only (laptop budget)
        plan.append((fin, "history_only"))
    for lab, fs in plan:
        if (EXPERIMENTS / f"summary__unseen_households__{lab}__{fs}.json").exists():
            log(f"unseen {lab} {fs} already done - skipped")
            continue
        if True:
            key, est, is_stack = build(lab, fs)
            cols = model_input_columns(fs)
            preds = []
            t0 = time.time()
            for i, (tr_mask, te_mask) in enumerate(unseen_household_folds(train, test), start=1):
                tr, te = train[tr_mask], test[te_mask]
                if is_stack:
                    est_i = TimeOrderedStackingRegressor(est.base_models).fit(tr[cols], tr.consumption.to_numpy(), tr.month.to_numpy())
                    del est_i.fold_bases_, est_i.oof_          # free memory (only needed for CV diagnostics)
                else:
                    from sklearn.base import clone
                    est_i = clone(est).fit(tr[cols], tr.consumption.to_numpy())
                preds.append(rolling_test_predict(est_i, te, fs).assign(group_fold=i))
            p = pd.concat(preds, ignore_index=True)
            m = regression_metrics(p.actual, p.predicted)
            name, fam = MODEL_INFO.get(key, (key, ""))
            summ = {"experiment": "unseen_households", "model_key": key, "label": lab, "model": name, "family": fam,
                    "feature_set": fs, "test_MAE": m["MAE"], "test_RMSE": m["RMSE"], "test_R2": m["R2"],
                    "test_n": m["n"], "n_group_folds": 5}
            tag = f"unseen_households__{lab}__{fs}"
            p.to_parquet(EXPERIMENTS / f"testpred__{tag}.parquet", index=False)
            (EXPERIMENTS / f"summary__{tag}.json").write_text(json.dumps(summ, indent=2))
            log(f"unseen       {lab:18s} {fs:20s} test_MAE {m['MAE']:.3f} test_RMSE {m['RMSE']:.2f} ({time.time()-t0:.0f}s)")


def stage_robustness():
    F, train, test = load_modelling_data()
    fin = final_label()
    # R1: exclude the 34 households (15 removed already + 19 low-activity) from training and test
    keep_tr, keep_te = ~train.is_low_activity_household, ~test.is_low_activity_household
    for lab in ["naive_last_month", fin]:
        run(lab, "robust_R1_excl34", train=train[keep_tr].reset_index(drop=True), test=test[keep_te].reset_index(drop=True))
    # R2: drop strictly isolated spikes from TRAINING targets. The flag is recomputed using
    # training-period months only, so the test months cannot influence which rows are dropped.
    monthly = pd.read_parquet(ROOT / "data" / "processed" / "monthly_clean.parquet")
    tr_only = monthly[monthly.month <= TRAIN_END]
    spikes = tr_only.loc[flag_isolated_spikes(tr_only), ["household_ID", "month"]]
    key = set(zip(spikes.household_ID, spikes.month))
    drop = np.array([(h, m) in key for h, m in zip(train.household_ID, train.month)])
    log(f"R2: dropping {drop.sum()} isolated-spike training rows (flag computed on training months only)")
    run(fin, "robust_R2_nospikes", train=train[~drop].reset_index(drop=True), test=test)


def stage_diagnostic():
    """POST-HOC diagnostic, added after test results showed a -19 kWh bias in October 2024.

    Hypothesis: with only ONE October (2023, an unusually low month) in the training data, the
    calendar-month features and lag_12 teach the models a spurious "October is low" effect.
    Re-run two tuned models without these features. Reported separately; NOT used to select or
    replace the final model (that would use the test set for model selection).
    """
    for fs in ["diag_no_calendar", "diag_no_calendar_no_lag12"]:
        for lab in ["random_forest", "gradient_boosting"]:
            run(lab, "posthoc_diagnostic", fs)


if __name__ == "__main__":
    stages = {"tuned": stage_tuned, "ablation": stage_ablation, "unseen": stage_unseen, "robustness": stage_robustness,
              "diagnostic": stage_diagnostic}
    for st in sys.argv[1:] or ["tuned", "ablation", "unseen", "robustness"]:
        log(f"=== stage {st} ===")
        stages[st]()
