import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, GRAY, NAVY, ORANGE, TEAL, load_household_features, load_importance,
                              load_metadata, load_monthly, load_results, load_test_predictions,
                              missing_artifacts_message, pretty)

U = "kWh"


def metrics(a, p):
    e = p - a
    ss = ((a - a.mean()) ** 2).sum()
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "R2": 1 - (e ** 2).sum() / ss if ss else np.nan, "n": len(a)}


@st.cache_data(show_spinner=False)
def household_groups():
    """Groups defined from information available before the test period."""
    m, hf = load_monthly(), load_household_features()
    pre = m[(m.month <= "2024-05-01")].dropna(subset=["consumption"]).groupby("household_ID").consumption.mean()
    level = pd.qcut(pre, 5, labels=["Lowest 20%", "20–40%", "40–60%", "60–80%", "Highest 20%"])
    g = hf.set_index("household_ID")[["socio_economic_class", "meter_type", "house_type"]]
    g["Consumption level (training mean)"] = level
    return g


def render():
    st.title("🔍 Model Performance Analysis")
    try:
        meta, preds, imp, res = load_metadata(), load_test_predictions(), load_importance(), load_results()
    except Exception as e:  # noqa: BLE001
        missing_artifacts_message(e)
        return
    models = [c[5:] for c in preds.columns if c.startswith("pred_")]
    default = models.index(meta["model_label"]) if meta["model_label"] in models else 0
    c1, c2 = st.columns([2, 3])
    label = c1.selectbox("Model", models, index=default, format_func=pretty)
    c2.info("All results on this page are from the **untouched test period (Jun–Oct 2024)**, predicted one month ahead "
            "by a model trained on Jan 2023 – May 2024.")
    a, p = preds.actual.to_numpy(), preds[f"pred_{label}"].to_numpy()
    mm = metrics(a, p)
    nv = metrics(a, preds["pred_naive_last_month"].to_numpy())
    k = st.columns(4)
    k[0].metric(f"Test MAE ({U})", f"{mm['MAE']:.2f}", f"{mm['MAE'] - nv['MAE']:+.2f} vs naive", delta_color="inverse")
    k[1].metric(f"Test RMSE ({U})", f"{mm['RMSE']:.2f}", f"{mm['RMSE'] - nv['RMSE']:+.2f} vs naive", delta_color="inverse")
    k[2].metric("Test R²", f"{mm['R2']:.3f}", f"{mm['R2'] - nv['R2']:+.3f} vs naive")
    k[3].metric("Test household-months", f"{mm['n']:,}")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Actual vs predicted", "Residuals", "By month & group", "Feature importance",
                                      "Over-fitting", "Extra experiments"])
    with t1:
        d = preds.assign(pred=p).sample(min(6000, len(preds)), random_state=0)
        lim = float(np.quantile(np.r_[a, p], 0.995))
        fig = px.scatter(d, x="actual", y="pred", opacity=0.3, color_discrete_sequence=[BLUE],
                         labels={"actual": f"Actual bill ({U})", "pred": f"Predicted bill ({U})"})
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=ORANGE, width=2), name="Ideal prediction (y = x)")
        fig.update_layout(title=f"{pretty(label)}: actual vs predicted (random 6,000 test rows)", height=520,
                          xaxis_range=[0, lim], yaxis_range=[0, lim])
        st.plotly_chart(fig, width="stretch")
        st.caption("Points on the orange line are perfect predictions. Points far below/above it are households whose "
                   "bill jumped up/down compared with their history — the hardest cases for every model.")
    with t2:
        r = p - a
        c = st.columns(2)
        fig = px.histogram(x=np.clip(r, -150, 150), nbins=80, color_discrete_sequence=[TEAL])
        fig.add_vline(x=0, line_color=NAVY)
        fig.update_layout(title="Residual distribution (prediction − actual, clipped ±150)", xaxis_title=f"Residual ({U})",
                          yaxis_title="Household-months", height=400)
        c[0].plotly_chart(fig, width="stretch")
        d = pd.DataFrame({"pred": p, "res": r}).sample(min(6000, len(r)), random_state=1)
        fig = px.scatter(d, x="pred", y="res", opacity=0.3, color_discrete_sequence=[BLUE], log_x=True)
        fig.add_hline(y=0, line_color=ORANGE)
        fig.update_layout(title="Residuals vs predicted value", xaxis_title=f"Predicted bill ({U}, log)",
                          yaxis_title=f"Residual ({U})", height=400, yaxis_range=[-300, 300])
        c[1].plotly_chart(fig, width="stretch")
        ae = np.abs(r)
        st.markdown(f"- Median absolute error **{np.median(ae):.1f} {U}**, 90th percentile **{np.quantile(ae, .9):.1f} {U}**.\n"
                    f"- Mean residual **{r.mean():+.2f} {U}** ({'over' if r.mean() > 0 else 'under'}-prediction on average).\n"
                    f"- The worst 5% of rows cause **{np.sort(ae)[::-1][:int(len(ae) * .05)].sum() / ae.sum():.0%}** of total absolute error.")
    with t3:
        pm = preds.assign(pred=p).groupby("month").apply(lambda g: pd.Series(metrics(g.actual.to_numpy(), g.pred.to_numpy())), include_groups=False)
        pn = preds.groupby("month").apply(lambda g: pd.Series(metrics(g.actual.to_numpy(), g.pred_naive_last_month.to_numpy())), include_groups=False)
        fig = go.Figure()
        fig.add_scatter(x=pm.index, y=pm.MAE, name=pretty(label), mode="lines+markers", line_color=TEAL)
        fig.add_scatter(x=pn.index, y=pn.MAE, name="Naive: last month", mode="lines+markers", line_color=GRAY)
        fig.update_layout(title=f"Test MAE by month ({U})", xaxis_title="Billing month", yaxis_title=f"MAE ({U})", height=380)
        st.plotly_chart(fig, width="stretch")
        try:
            groups = household_groups()
            gsel = st.selectbox("Household group", list(groups.columns), index=3)
            d = preds.assign(pred=p).join(groups[gsel], on="household_ID").dropna(subset=[gsel])
            gm = d.groupby(gsel, observed=True).apply(
                lambda g: pd.Series({**metrics(g.actual.to_numpy(), g.pred.to_numpy()),
                                     "naive MAE": np.abs(g.pred_naive_last_month - g.actual).mean(),
                                     "mean bill": g.actual.mean()}), include_groups=False).reset_index()
            gm = gm[gm.n >= 100]
            fig = px.bar(gm, x=gsel, y="MAE", color_discrete_sequence=[TEAL], text=gm.MAE.round(1),
                         hover_data={"n": True, "mean bill": ":.0f"})
            fig.add_scatter(x=gm[gsel], y=gm["naive MAE"], mode="markers", marker=dict(color=NAVY, size=11, symbol="line-ew-open", line_width=3), name="naive MAE")
            fig.update_layout(title=f"Test MAE by {gsel.lower()} ({U}; groups with ≥100 rows)", xaxis_title="", yaxis_title=f"MAE ({U})", height=420)
            st.plotly_chart(fig, width="stretch")
            gm["MAE as % of mean bill"] = 100 * gm.MAE / gm["mean bill"]
            st.dataframe(gm.round(2), hide_index=True, width="stretch")
            st.caption("Absolute errors grow with the size of the bill: large households have larger kWh errors but "
                       "similar or smaller errors *relative* to their bill. Consumption level groups use the training-period mean.")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Household groups unavailable: {e}")
    with t4:
        pi = imp[imp.method.str.startswith("permutation")].sort_values("importance")
        fig = px.bar(pi, x="importance", y="feature", orientation="h", error_x="importance_std", color_discrete_sequence=[TEAL])
        fig.update_layout(title=f"Permutation importance — {pretty(meta['model_label'])} (increase in test MAE, {U}, when a feature is shuffled)",
                          xaxis_title=f"Increase in MAE ({U})", yaxis_title="", height=560)
        st.plotly_chart(fig, width="stretch")
        st.caption("Shuffling a feature breaks its link to the target; the larger the error increase, the more the model "
                   "relies on it. Correlated features (e.g. lag_1, roll_mean_3) share credit, so each may look less "
                   "important than the information it carries. Importance is not causation.")
        tree = imp[imp.method.str.startswith("built-in")]
        if not tree.empty:
            mod = st.selectbox("Built-in importance of a tree model", tree.model.unique(), format_func=pretty)
            d = tree[tree.model == mod].sort_values("importance").tail(15)
            fig = px.bar(d, x="importance", y="feature", orientation="h", color_discrete_sequence=[BLUE])
            fig.update_layout(title=f"{pretty(mod)}: built-in importance (top 15)", xaxis_title="Share of total importance", yaxis_title="", height=460)
            st.plotly_chart(fig, width="stretch")
    with t5:
        main = res[res.experiment.isin(["main", "tuned"]) & (res.feature_set == "history_only")].drop_duplicates("label", keep="last")
        main = main[main.label != "dummy_mean"].assign(Model=lambda d: d.label.map(pretty), gap=lambda d: d.cv_MAE - d.train_MAE)
        fig = go.Figure()
        fig.add_bar(y=main.Model, x=main.train_MAE, name="Training MAE", orientation="h", marker_color=TEAL)
        fig.add_bar(y=main.Model, x=main.cv_MAE, name="Validation MAE", orientation="h", marker_color=ORANGE)
        fig.update_layout(barmode="group", title=f"Training vs validation MAE ({U})", xaxis_title=f"MAE ({U})", height=480)
        st.plotly_chart(fig, width="stretch")
        st.markdown("A large gap between training and validation error means the model memorises training households "
                    "(over-fitting). Forests show the largest gaps but still generalise well because averaging many "
                    "trees cancels much of the memorised noise; tuning chose larger leaves to reduce it.")
    with t6:
        exp = res[~res.experiment.isin(["main", "untuned", "tuned"])].copy()
        if exp.empty:
            st.info("No extra experiments saved.")
        else:
            exp["Model"] = exp.label.map(pretty)
            names = {"ablation": "History + Wave 1 survey (vs history-only)", "unseen_households": "Unseen households (scenario B)",
                     "robust_R1_excl34": "R1: excluding 34 always-zero / low-activity households",
                     "robust_R2_nospikes": "R2: isolated spikes removed from training",
                     "posthoc_diagnostic": "POST-HOC diagnostic (not used for selection): without calendar features"}
            exp["Experiment"] = exp.experiment.map(names).fillna(exp.experiment)
            st.dataframe(exp[["Experiment", "Model", "feature_set", "cv_MAE", "test_MAE", "test_RMSE", "test_R2", "test_n"]]
                         .rename(columns={"feature_set": "Features", "cv_MAE": f"Val MAE ({U})", "test_MAE": f"Test MAE ({U})",
                                          "test_RMSE": f"Test RMSE ({U})", "test_R2": "Test R²", "test_n": "Test rows"}).round(3),
                         hide_index=True, width="stretch")
            st.caption("Compare with the main results (history-only, all households) on the Model Comparison page. "
                       "Survey features are used only after each household's interview date, so they are available for "
                       "just 16% of training rows.")
            st.markdown("""
**Key findings**
- **Survey information:** adding Wave 1 features changes the ensembles' MAE by only about ±0.1 kWh, so bill history already carries the information.
- **Unseen households:** errors are almost the same as for known households (Voting 19.79 vs 19.67 kWh), as long as the household has a billing history.
- **Robustness:** removing the 34 low-activity households or the isolated spikes does not change the conclusions.
- **Post-hoc October finding:** without the calendar-month features, test MAE drops to **18.16 kWh** (Gradient Boosting), but validation MAE gets worse. The training data contain only one October, which was unusually low, so the models learned a wrong "October is low" pattern. This was found *after* testing, so it is reported as a limitation and the final model was not changed.
""")
