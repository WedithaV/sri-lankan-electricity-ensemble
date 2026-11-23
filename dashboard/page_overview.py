import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (TEAL, BLUE, GRAY, card, kpi, load_metadata, load_monthly, load_results,
                              missing_artifacts_message, pretty)


def render():
    st.markdown("""<div class="hero"><h1>Sri Lankan Household Electricity Consumption Prediction</h1>
    <p>Predicting next month's electricity bill (kWh) of Sri Lankan households with <b>Bagging</b>, <b>Boosting</b>
    and <b>Stacking</b> ensemble learning — and checking honestly whether they beat simple forecasts.</p></div>""",
                unsafe_allow_html=True)
    try:
        meta, res, monthly = load_metadata(), load_results(), load_monthly()
    except Exception as e:  # noqa: BLE001
        missing_artifacts_message(e)
        return
    u = meta["unit"]
    main = res[res.experiment.isin(["main", "untuned", "tuned"])]

    c = st.columns(4)
    with c[0]:
        kpi("Monthly observations", f"{monthly.consumption.notna().sum():,}", "household-month bills (after cleaning)")
    with c[1]:
        kpi("Households", f"{monthly.household_ID.nunique():,}", "LECO customers in the survey panel")
    with c[2]:
        kpi("Features used", f"{len(meta['feature_columns'])}", "bill history + calendar (final model)")
    with c[3]:
        kpi("Models evaluated", f"{main.label.nunique()}", "baselines, bagging, boosting, voting, stacking")
    st.write("")
    c = st.columns(4)
    with c[0]:
        kpi("Best model", pretty(meta["model_label"]), f"{meta['family']} · selected by validation MAE", accent=True)
    with c[1]:
        kpi("Test MAE", f"{meta['test_MAE']:.2f} {u}", "average absolute error per household-month", accent=True)
    with c[2]:
        kpi("Test RMSE", f"{meta['test_RMSE']:.2f} {u}", "penalises large errors more", accent=True)
    with c[3]:
        kpi("Test R²", f"{meta['test_R2']:.3f}", "share of bill variation explained", accent=True)
    st.write("")
    p0, p1 = meta["training_period"], meta["test_period"]
    st.info(f"**Observation period:** billing months Oct 2022 – Oct 2024 (25 months). "
            f"Models are trained on {p0[0]} – {p0[1]} and evaluated on the untouched test period {p1[0]} – {p1[1]}, "
            f"predicting **one month ahead** using only bills that were already known.")

    left, right = st.columns([3, 2])
    with left:
        g = monthly.dropna(subset=["consumption"]).groupby("month").consumption
        fig = go.Figure()
        fig.add_scatter(x=g.median().index, y=g.median().values, name="Median household", mode="lines+markers", line_color=TEAL)
        fig.add_scatter(x=g.mean().index, y=g.mean().values, name="Mean household", mode="lines+markers", line_color=BLUE)
        fig.add_vrect(x0=p1[0] + "-01", x1="2024-10-20", fillcolor=GRAY, opacity=0.15, line_width=0,
                      annotation_text="test period", annotation_position="top left")
        fig.update_layout(title=f"Monthly household consumption ({u})", yaxis_title=f"{u} per month",
                          xaxis_title="Billing month", height=360)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### About the data")
        st.markdown(
            "**RECON-SL** (LIRNEasia with Lanka Electricity Company, LECO) links monthly electricity bills, smart-meter "
            "readings and three household survey waves for about 4,000 households.\n\n"
            "**Geographical scope:** households in **LECO's service areas along Sri Lanka's western and south-western "
            "coast** (e.g. Moratuwa, Panadura, Kalutara, Galle, Negombo, Nugegoda). The results describe this sample — "
            "they are **not** representative of every household in Sri Lanka.")
        st.caption("Source: RECON-SL, DOI 10.21227/n1dk-q860 · Zenodo 15023048 · CC-BY-4.0")

    st.markdown("### What is ensemble learning?")
    st.caption("Combining many models usually gives more reliable predictions than any single one.")
    c = st.columns(3)
    with c[0]:
        card("🌳 Bagging", "Many decision trees are trained on random samples of households and their predictions are "
             "<b>averaged</b> (Random Forest, Extra Trees). Averaging cancels out individual trees' mistakes.")
    with c[1]:
        card("📈 Boosting", "Small trees are added <b>one after another</b>; each new tree learns to correct the errors "
             "left by the previous ones (Gradient Boosting, XGBoost).")
    with c[2]:
        card("🧩 Stacking & Voting", "Different models are combined. Voting takes a simple average; stacking trains a "
             "<b>meta-model</b> that learns how much to trust each model — here using only past months.")
