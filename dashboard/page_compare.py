import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (FAMILY_COLORS, NAVY, load_cv, load_metadata, load_results, missing_artifacts_message,
                              pretty)

U = "kWh"
BASELINES = ["naive_last_month", "naive_mean_3m", "dummy_mean", "linear", "ridge"]


def comparison_table(res: pd.DataFrame, version: str) -> pd.DataFrame:
    base = res[(res.experiment == "main") & (res.feature_set == "history_only")]
    ens = res[(res.experiment == version) & (res.feature_set == "history_only")]
    t = pd.concat([base, ens]).drop_duplicates("label", keep="last").copy()
    t["Model"] = t.label.map(pretty)
    t["version"] = t.experiment.map({"main": "baseline", "untuned": "untuned", "tuned": "tuned"})
    return t.sort_values("cv_MAE").reset_index(drop=True)


def render():
    st.title("🏆 Ensemble Model Comparison")
    try:
        res, cv, meta = load_results(), load_cv(), load_metadata()
    except Exception as e:  # noqa: BLE001
        missing_artifacts_message(e)
        return
    st.markdown("Every model uses **the same features, the same training period, the same 5 time-ordered validation "
                "folds and the same untouched test period (Jun–Oct 2024)**. Models are ranked by **validation MAE** — "
                "the pre-registered selection metric; the test set was never used to choose a model.")
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    version = c1.radio("Ensemble versions", ["tuned", "untuned"], horizontal=True,
                       help="Tuned = hyper-parameters (and loss function) chosen by RandomizedSearchCV on the validation folds.")
    hide_dummy = c2.toggle("Hide DummyRegressor", value=True, help="Its error is ~4× larger and squashes the charts.")
    families = c3.multiselect("Model families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    t = comparison_table(res, version)
    if hide_dummy:
        t = t[t.label != "dummy_mean"]
    t = t[t.family.isin(families)]
    if t.empty:
        st.warning("Select at least one family.")
        return

    best = t.iloc[0]
    naive = t.set_index("label").loc["naive_last_month"] if "naive_last_month" in set(t.label) else None
    k = st.columns(4)
    k[0].metric("Best by validation MAE", pretty(best.label))
    k[1].metric(f"Validation MAE ({U})", f"{best.cv_MAE:.2f}",
                f"{best.cv_MAE - naive.cv_MAE:+.2f} vs naive" if naive is not None else None, delta_color="inverse")
    k[2].metric(f"Test MAE ({U})", f"{best.test_MAE:.2f}",
                f"{best.test_MAE - naive.test_MAE:+.2f} vs naive" if naive is not None else None, delta_color="inverse")
    k[3].metric(f"Test RMSE ({U})", f"{best.test_RMSE:.2f}",
                f"{best.test_RMSE - naive.test_RMSE:+.2f} vs naive" if naive is not None else None, delta_color="inverse")

    show = t[["Model", "family", "version", "cv_MAE", "cv_MAE_std", "cv_RMSE", "cv_R2", "test_MAE", "test_RMSE",
              "test_R2", "train_MAE", "fit_seconds_full"]].rename(columns={
        "family": "Family", "version": "Version", "cv_MAE": f"Val MAE ({U})", "cv_MAE_std": "Val MAE std",
        "cv_RMSE": f"Val RMSE ({U})", "cv_R2": "Val R²", "test_MAE": f"Test MAE ({U})", "test_RMSE": f"Test RMSE ({U})",
        "test_R2": "Test R²", "train_MAE": f"Train MAE ({U})", "fit_seconds_full": "Training time (s)"})
    show.insert(0, "Rank", range(1, len(show) + 1))
    st.dataframe(show.style.format(precision=2).highlight_min(subset=[f"Val MAE ({U})", f"Test MAE ({U})", f"Test RMSE ({U})"],
                                                               color="#D5F0EC").highlight_max(subset=["Test R²"], color="#D5F0EC"),
                 hide_index=True, width="stretch")
    st.caption(f"Final saved model: **{pretty(meta['model_label'])}** (rule: {meta['selection_rule']}).")

    tabs = st.tabs(["MAE", "RMSE", "R²", "Validation folds", "Training time"])
    order = t.Model.tolist()

    def bars(metric_val, metric_test, title, lower_better=True):
        d = t.melt(id_vars=["Model", "family"], value_vars=[metric_val, metric_test], var_name="set", value_name="value")
        d["set"] = d["set"].map({metric_val: "Validation (mean of 5 folds)", metric_test: "Test (Jun–Oct 2024)"})
        fig = px.bar(d, y="Model", x="value", color="set", barmode="group", orientation="h",
                     category_orders={"Model": order}, color_discrete_sequence=["#0F9D8F", "#2A78D6"],
                     text=d["value"].round(2))
        if naive is not None:
            fig.add_vline(x=naive[metric_test], line_dash="dash", line_color=NAVY,
                          annotation_text="naive (test)", annotation_position="top")
        lo = d.value.min() * (0.9 if lower_better else 0.97)
        fig.update_layout(title=title, xaxis_title=title, yaxis_title="", height=90 + 38 * len(order), legend_title="",
                          xaxis_range=[lo, d.value.max() * 1.04])
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("The x-axis does not start at zero so that small differences are visible.")

    with tabs[0]:
        bars("cv_MAE", "test_MAE", f"MAE ({U}) — lower is better")
        st.markdown("**MAE** is the average absolute difference between the predicted and the actual bill: "
                    "an MAE of 19 kWh means predictions are on average 19 kWh away from the real bill.")
    with tabs[1]:
        bars("cv_RMSE", "test_RMSE", f"RMSE ({U}) — lower is better")
        st.markdown("**RMSE** squares errors before averaging, so one 200 kWh miss counts as much as 100 misses of 20 kWh. "
                    "Models trained with squared error (Ridge, untuned forests) are strong on RMSE; MAE-oriented losses "
                    "(absolute / Huber) are strong on MAE.")
    with tabs[2]:
        bars("cv_R2", "test_R2", "R² — higher is better", lower_better=False)
    with tabs[3]:
        exp_map = {l: ("main" if l in BASELINES else version) for l in t.label}
        cvf = cv[cv.apply(lambda r: r.label in exp_map and r.experiment == exp_map[r.label] and r.feature_set == "history_only", axis=1)].copy()
        cvf["Model"] = cvf.label.map(pretty)
        pick = st.multiselect("Models", order, default=[m for m in order if m in (pretty(best.label), "Naive: last month", "Ridge Regression", "Voting", "Stacking")][:5])
        fig = px.line(cvf[cvf.Model.isin(pick)], x="validation_month", y="val_MAE", color="Model", markers=True)
        fig.update_layout(title=f"Validation MAE per fold ({U}) — each fold trains on earlier months only",
                          xaxis_title="Validation month", yaxis_title=f"MAE ({U})", height=420, legend_title="")
        st.plotly_chart(fig, width="stretch")
    with tabs[4]:
        tt = t[t.fit_seconds_full > 0.01]
        fig = px.bar(tt, y="Model", x="fit_seconds_full", color="family", orientation="h", log_x=True,
                     color_discrete_map=FAMILY_COLORS, category_orders={"Model": order})
        fig.update_layout(title="Training time on the full training period (seconds, log scale)", xaxis_title="seconds",
                          yaxis_title="", height=420, legend_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Measured on a 12-core laptop. Stacking includes the time-ordered out-of-fold fits of its base models.")

    st.markdown("### Why do the ensembles behave differently?")
    st.markdown("""
- **Persistence is a very strong baseline.** About 85% of the variation in bills is *between* households, and a
  household's bill this month is very close to last month's (correlation ≈ 0.93). Every model mostly learns to repeat
  the recent level; the remaining gains come from households that change level.
- **Bagging (Random Forest, Extra Trees)** averages many deep trees → low variance and the best RMSE among single
  models, but its squared-error splits target large errors rather than typical (MAE) errors.
- **Boosting (Gradient Boosting, XGBoost)** gained most from tuning, once the search could choose an MAE-friendly
  loss (Huber / absolute error) and a log target; untuned XGBoost over-fitted with its default depth and learning rate.
- **Voting and Stacking** combine members whose predictions are ~98–99% correlated, so the gains over the best single
  model are small. The stacking meta-learner is trained only on genuinely forward-in-time predictions.
""")
    st.markdown("### Why are test errors higher than validation errors for the ensembles?")
    st.markdown("""
- On the test period **no model is significantly better than "same as last month" on MAE**. In a household-level
  bootstrap, Voting, Stacking and Random Forest are statistically tied with it. The ensembles do make clearly fewer
  **large** errors (RMSE about 44 vs 49 kWh).
- Most of the gap comes from **October 2024**. The training data contain only one October (2023), which was unusually
  low, so the calendar-month features taught the tree models "October is low". In October 2024 that pattern did not
  repeat, and the models under-predicted by about 16–19 kWh on average.
- Validation months (Jan–May 2024) could not reveal this. The lesson: with about two years of data, seasonal
  patterns cannot be learned reliably. Details are on the *Model Performance → Extra experiments* tab.
""")
