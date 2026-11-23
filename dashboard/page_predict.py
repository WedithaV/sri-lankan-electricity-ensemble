import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, GRAY, MODELS, NAVY, ORANGE, TEAL, load_bundle, load_metadata, load_test_predictions,
                              missing_artifacts_message, pretty)

U = "kWh"
N_MONTHS = 12
EXAMPLES = {   # synthetic illustrative households (most recent bill first) -- not real customers
    "Typical stable household": [96, 92, 99, 101, 94, 90, 97, 95, 99, 93, 91, 96],
    "Household with air conditioning": [265, 248, 280, 301, 290, 255, 240, 236, 262, 278, 270, 251],
    "Recent increase in usage": [180, 175, 120, 115, 118, 110, 112, 109, 115, 111, 108, 113],
    "Some bills missing": [88, None, 85, 90, None, 86, 84, 89, 91, 87, None, 85],
}


def month_list(last: dt.date):
    last = pd.Timestamp(last).to_period("M").to_timestamp()
    return [last - pd.DateOffset(months=i) for i in range(N_MONTHS)]


@st.cache_data(show_spinner=False)
def error_bands(label: str):
    """Empirical test-period error of the final model, by size of the prediction."""
    p = load_test_predictions()
    d = pd.DataFrame({"pred": p[f"pred_{label}"], "res": p.actual - p[f"pred_{label}"]})
    d["bin"] = pd.qcut(d.pred, 10, duplicates="drop")
    return d.groupby("bin", observed=True).res.agg(lo=lambda s: s.quantile(.1), hi=lambda s: s.quantile(.9),
                                                   mae=lambda s: s.abs().mean(), n="size").reset_index()


def render():
    st.title("⚡ Electricity Consumption Prediction")
    try:
        bundle, meta = load_bundle(), load_metadata()
    except Exception as e:  # noqa: BLE001
        missing_artifacts_message(e)
        return
    from src.predict import InputError, predict_next_month

    st.markdown(f"""
The model (**{pretty(meta['model_label'])}**) predicts a household's **next monthly bill** from its **own recent bills**.
It was built for existing electricity customers who already have a billing history — the same situation as in the
evaluation. Enter the most recent bills (in **{U}**, as printed on the electricity bill).
Leave a box **empty if that bill is not available** — an empty box is treated as *missing*, never as zero.
""")
    st.caption("Household survey information is not needed: the final model uses only bill history and the calendar "
               "month, because survey answers added no reliable accuracy under leakage-free conditions "
               "(see Model Performance → Extra experiments).")

    def fill(values):
        for i in range(N_MONTHS):
            v = values[i] if i < len(values) else None
            st.session_state[f"bill_{i}"] = None if v is None else float(v)

    if "bill_0" not in st.session_state:            # first visit: start with the typical example
        fill(EXAMPLES["Typical stable household"])
    ex = st.columns(len(EXAMPLES))
    for col, (name, vals) in zip(ex, EXAMPLES.items()):
        col.button(name, width="stretch", on_click=fill, args=(vals,))
    st.caption("Example buttons fill in illustrative (synthetic) values.")

    with st.form("predict_form"):
        last = st.date_input("Month of the most recent bill", value=dt.date(2024, 10, 1),
                             min_value=dt.date(2020, 1, 1), max_value=dt.date(2030, 12, 1),
                             help="The prediction is for the following month.")
        months = month_list(last)
        st.markdown(f"**Bills for the last {N_MONTHS} months ({U})** — most recent first")
        bills = []
        for r in range(0, N_MONTHS, 6):
            cols = st.columns(6)
            for j, c in enumerate(cols):
                i = r + j
                # value=None is essential: without it Streamlit falls back to min_value (0) when a box
                # is cleared, which would silently turn a MISSING bill into a 0 kWh bill.
                bills.append(c.number_input(months[i].strftime("%b %Y") + (" (latest)" if i == 0 else ""),
                                            min_value=0.0, max_value=10000.0, value=None, step=1.0, format="%.0f",
                                            key=f"bill_{i}", placeholder="missing"))
        submitted = st.form_submit_button("⚡ Predict Monthly Electricity Consumption", type="primary", width="stretch")

    if not submitted:
        return
    history = {m: b for m, b in zip(months, bills)}
    if bills[0] is None:
        st.error("Please enter at least the most recent bill (the first box). It is the most important input.")
        return
    try:
        out = predict_next_month(bundle, history)
    except InputError as e:
        st.error(str(e))
        return
    pred, target = out["prediction_kwh"], out["month"]
    if pd.Timestamp(target) > pd.Timestamp("2025-12-01") or pd.Timestamp(target) < pd.Timestamp("2023-01-01"):
        st.warning("This month is outside the period the model was trained and tested on (2023–2024); "
                   "seasonal patterns may differ.")

    bands = error_bands(meta["model_label"])
    row = bands[bands.bin.apply(lambda b: b.left < pred <= b.right)]
    row = row.iloc[0] if len(row) else bands.iloc[-1 if pred > bands.bin.iloc[-1].right else 0]
    lo, hi = max(pred + row.lo, 0), pred + row.hi

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"""<div class="result"><div class="note">Estimated Monthly Electricity Consumption</div>
        <div class="big">{pred:,.2f} {U}</div><div class="note">for <b>{target:%B %Y}</b></div>
        <div class="note" style="margin-top:10px">Typical range: <b>{lo:,.0f} – {hi:,.0f} {U}</b><br>
        (80% of similar test-period predictions were within this range)</div></div>""", unsafe_allow_html=True)
        comp = {"Naive: same as last month": float(bills[0])}
        ridge_path = MODELS / "ridge_baseline.joblib"
        if ridge_path.exists():
            comp["Ridge Regression"] = predict_next_month(load_bundle("ridge_baseline.joblib"), history)["prediction_kwh"]
        st.markdown("**Other models for comparison**")
        st.dataframe(pd.DataFrame({"Model": list(comp), f"Prediction ({U})": [round(v, 2) for v in comp.values()]}),
                     hide_index=True, width="stretch")
    with c2:
        h = pd.Series(history).sort_index()
        fig = go.Figure()
        fig.add_scatter(x=h.index, y=h.values, mode="lines+markers", name="Entered bills", line_color=TEAL, connectgaps=False)
        fig.add_scatter(x=[target, target], y=[lo, hi], mode="lines", line=dict(color=GRAY, width=8), name="Typical range", opacity=0.5)
        fig.add_scatter(x=[target], y=[pred], mode="markers", marker=dict(color=ORANGE, size=14, symbol="diamond"),
                        name="Prediction")
        fig.update_layout(title="Bill history and next-month prediction", xaxis_title="Billing month",
                          yaxis_title=f"{U} per month", height=380)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Inputs the model actually used (computed from the entered bills)"):
        f = pd.Series(out["features"]).rename("value").to_frame()
        f["meaning"] = [meta["feature_descriptions"].get(k, "") for k in f.index]
        st.dataframe(f, width="stretch")
    st.warning("**Disclaimer.** This is an estimate based on historical bills of households in LECO service areas "
               "(2022–2024). It does not guarantee actual future consumption, which can change with weather, new "
               "appliances, occupancy or tariff changes. It is a university project, not an official LECO/CEB service.")
