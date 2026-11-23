import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (BLUE, GRAY, ORANGE, TEAL, load_household_features, load_hourly_profile, load_monthly,
                              missing_artifacts_message)

U = "kWh"
CHARACTERISTICS = {
    "Household size": "household_size", "Socio-economic class": "socio_economic_class",
    "House type": "house_type", "Number of air conditioners": "n_ac", "Number of refrigerators": "n_refrigerator",
    "Number of fans": "n_fans", "Washing machine": "n_washing_machine", "Solar panels": "has_solar",
    "Meter type": "meter_type", "Tenure": "tenure",
}


@st.cache_data(show_spinner=False)
def household_table():
    m, hf = load_monthly(), load_household_features()
    hm = m.dropna(subset=["consumption"]).groupby("household_ID").consumption.mean().rename("mean_kwh")
    H = hf.merge(hm, left_on="household_ID", right_index=True)
    # anonymous display index instead of household identifiers
    H["Household"] = "H" + pd.Series(np.arange(1, len(H) + 1), index=H.index).astype(str).str.zfill(4)
    return H


def render():
    st.title("📊 Dataset Exploration")
    st.caption("Monthly billing panel (cleaned) and Wave 1 household characteristics. Household identifiers are not "
               "shown. All consumption values are in kWh per month.")
    try:
        m, H = load_monthly(), household_table()
    except Exception as e:  # noqa: BLE001
        missing_artifacts_message(e)
        return

    with st.expander("🔎 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        areas = sorted(H.csc_area.dropna().unique())
        sel_area = f1.multiselect("LECO CSC area", areas, placeholder="All areas")
        sel_sec = f2.multiselect("Socio-economic class", sorted(H.socio_economic_class.unique()), placeholder="All classes")
        months = sorted(m.month.unique())
        rng = f3.select_slider("Billing months", options=months, value=(months[0], months[-1]),
                               format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
    Hf = H.copy()
    if sel_area:
        Hf = Hf[Hf.csc_area.isin(sel_area)]
    if sel_sec:
        Hf = Hf[Hf.socio_economic_class.isin(sel_sec)]
    mf = m[m.household_ID.isin(Hf.household_ID) & m.month.between(rng[0], rng[1])]
    obs = mf.dropna(subset=["consumption"])
    if obs.empty:
        st.warning("No data for this filter combination.")
        return

    c = st.columns(5)
    c[0].metric("Households", f"{mf.household_ID.nunique():,}")
    c[1].metric("Observed bills", f"{len(obs):,}")
    c[2].metric(f"Median bill ({U})", f"{obs.consumption.median():.0f}")
    c[3].metric(f"Mean bill ({U})", f"{obs.consumption.mean():.0f}")
    c[4].metric("Missing bills", f"{mf.consumption.isna().mean():.2%}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Consumption", "Household patterns", "Household characteristics",
                                            "Smart-meter (15-min)", "Data preview & quality"])
    with tab1:
        a, b = st.columns(2)
        x = obs.consumption
        fig = px.histogram(obs.assign(c=x.clip(upper=x.quantile(.99))), x="c", nbins=60, color_discrete_sequence=[TEAL])
        fig.update_layout(title="Distribution of monthly bills (capped at 99th percentile)",
                          xaxis_title=f"Monthly consumption ({U})", yaxis_title="Household-months", height=360)
        a.plotly_chart(fig, width="stretch")
        g = obs.groupby("month").consumption
        st_ = pd.DataFrame({"median": g.median(), "q25": g.quantile(.25), "q75": g.quantile(.75), "mean": g.mean()})
        fig = go.Figure()
        fig.add_scatter(x=st_.index, y=st_.q75, line_width=0, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=st_.index, y=st_.q25, fill="tonexty", fillcolor="rgba(15,157,143,0.12)", line_width=0,
                        name="Interquartile range")
        fig.add_scatter(x=st_.index, y=st_["median"], name="Median", line_color=TEAL, mode="lines+markers")
        fig.add_scatter(x=st_.index, y=st_["mean"], name="Mean", line_color=BLUE, mode="lines+markers")
        fig.update_layout(title="Monthly consumption trend", xaxis_title="Billing month", yaxis_title=f"{U} per month", height=360)
        b.plotly_chart(fig, width="stretch")
        st.caption("Bills are right-skewed: a minority of large users pull the mean above the median. "
                   "The billing month mostly reflects electricity used in the previous calendar month.")

    with tab2:
        hh = obs.groupby("household_ID").consumption.agg(["mean", "std"])
        a, b = st.columns(2)
        fig = px.histogram(hh.assign(m=hh["mean"].clip(upper=hh["mean"].quantile(.99))), x="m", nbins=60,
                           color_discrete_sequence=[BLUE])
        fig.update_layout(title="Average monthly consumption per household", xaxis_title=f"Household mean ({U}/month)",
                          yaxis_title="Households", height=360)
        a.plotly_chart(fig, width="stretch")
        n = b.slider("Households to display", 5, 40, 15)
        sample = pd.Series(obs.household_ID.unique()).sample(min(n, obs.household_ID.nunique()), random_state=1)
        piv = obs[obs.household_ID.isin(sample)].pivot(index="month", columns="household_ID", values="consumption")
        fig = go.Figure()
        for i, col in enumerate(piv.columns):
            fig.add_scatter(x=piv.index, y=piv[col], mode="lines", line=dict(color=GRAY, width=1), opacity=0.6,
                            name=f"Household {i + 1}", showlegend=False)
        fig.add_scatter(x=g.median().index, y=g.median().values, mode="lines", line=dict(color=TEAL, width=3), name="Median of filtered households")
        fig.update_layout(title="Individual households over time (random sample)", xaxis_title="Billing month",
                          yaxis_title=f"{U} per month", yaxis_type="log", height=360)
        b.plotly_chart(fig, width="stretch")
        prev = mf.sort_values(["household_ID", "month"]).assign(prev=lambda d: d.groupby("household_ID").consumption.shift(1)).dropna(subset=["consumption", "prev"])
        prev = prev[(prev.consumption > 0) & (prev.prev > 0)].sample(min(8000, len(prev)), random_state=0)
        fig = px.scatter(prev, x="prev", y="consumption", opacity=0.25, log_x=True, log_y=True, color_discrete_sequence=[BLUE])
        fig.add_scatter(x=[1, 3000], y=[1, 3000], mode="lines", line=dict(color=ORANGE), name="same as last month")
        fig.update_layout(title="This month vs last month (why history is so predictive)",
                          xaxis_title=f"Previous month ({U}, log)", yaxis_title=f"Current month ({U}, log)", height=420)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        a, b = st.columns([1, 2])
        choice = a.selectbox("Characteristic (Wave 1 survey)", list(CHARACTERISTICS))
        col = CHARACTERISTICS[choice]
        d = Hf[[col, "mean_kwh"]].dropna().copy()
        if pd.api.types.is_numeric_dtype(d[col]) and d[col].nunique() > 2:
            d[col] = d[col].clip(upper=d[col].quantile(.98)).round().astype(int).astype(str)
            order = sorted(d[col].unique(), key=lambda v: float(v))
        else:
            d[col] = d[col].astype(str)
            order = sorted(d[col].unique())
        counts = d[col].value_counts()
        order = [o for o in order if counts.get(o, 0) >= 15]
        fig = px.box(d[d[col].isin(order)], x=col, y="mean_kwh", points=False, category_orders={col: order},
                     color_discrete_sequence=[TEAL])
        fig.update_layout(title=f"Average consumption by {choice.lower()}", xaxis_title=choice,
                          yaxis_title=f"Mean consumption ({U}/month)", height=420)
        fig.update_yaxes(range=[0, d.mean_kwh.quantile(.97)])
        b.plotly_chart(fig, width="stretch")
        a.caption("Groups with fewer than 15 households are hidden. Associations are descriptive, not causal: "
                  "households with more appliances also tend to be larger and wealthier.")
        own = {"Refrigerator": "n_refrigerator", "TV": "n_tv", "Washing machine": "n_washing_machine",
               "Laptop": "n_laptop", "Electric kettle": "n_kettle", "Air conditioner": "n_ac", "Electric oven": "n_oven",
               "Water pump": "n_water_pump", "Microwave": "n_microwave", "Electric geyser": "n_geyser", "Solar panels": "has_solar"}
        rates = pd.Series({k: (Hf[v] > 0).mean() for k, v in own.items()}).sort_values()
        fig = px.bar(x=rates.values, y=rates.index, orientation="h", color_discrete_sequence=[TEAL], text=[f"{v:.0%}" for v in rates.values])
        fig.update_layout(title="Appliance ownership (share of households)", xaxis_title="Share owning at least one",
                          yaxis_title="", xaxis_tickformat=".0%", height=420)
        a2, b2 = st.columns(2)
        a2.plotly_chart(fig, width="stretch")
        area = Hf.groupby("csc_area").mean_kwh.agg(["median", "size"]).query("size >= 30").sort_values("median")
        fig = px.bar(x=area["median"], y=area.index.str.title(), orientation="h", color_discrete_sequence=[BLUE],
                     hover_data={"households": area["size"].values})
        fig.update_layout(title="Median consumption by LECO CSC area (≥30 households)", xaxis_title=f"{U}/month",
                          yaxis_title="", height=520)
        b2.plotly_chart(fig, width="stretch")

    with tab4:
        try:
            prof = load_hourly_profile()
        except Exception as e:  # noqa: BLE001
            missing_artifacts_message(e)
        else:
            pp = prof[prof.n_intervals >= 40].groupby(["day_type", "hour"]).kwh_per_hour.median().reset_index()
            fig = px.line(pp, x="hour", y="kwh_per_hour", color="day_type", markers=True,
                          color_discrete_map={"Weekday": TEAL, "Weekend": ORANGE})
            fig.add_vrect(x0=18.5, x1=22.5, fillcolor=GRAY, opacity=0.15, line_width=0,
                          annotation_text="time-of-use peak 18:30–22:30", annotation_position="top left")
            fig.update_layout(title="Electricity use by hour of day — median household (15-minute smart-meter data)",
                              xaxis_title="Hour of day", yaxis_title=f"{U} per hour", height=420, legend_title="")
            st.plotly_chart(fig, width="stretch")
            st.caption("Interval energy = difference of the cumulative meter register between readings exactly 15 minutes "
                       "apart; gaps are never treated as zero. Smart-meter data cover about a third of the panel "
                       "households and are used for analysis only, not as model inputs.")

    with tab5:
        a, b = st.columns([3, 2])
        ids = {h: f"H{i:04d}" for i, h in enumerate(sorted(mf.household_ID.unique()), start=1)}
        preview = mf.assign(Household=mf.household_ID.map(ids), Month=mf.month.dt.strftime("%Y-%m"))[
            ["Household", "Month", "consumption", "is_missing", "is_zero", "is_isolated_spike"]].rename(
            columns={"consumption": f"Consumption ({U})"})
        a.dataframe(preview.head(500), hide_index=True, width="stretch", height=380)
        a.caption("First 500 rows of the filtered panel (anonymised household labels).")
        miss = mf.groupby("month").agg(missing=("is_missing", "sum"), zero=("is_zero", "sum"))
        fig = go.Figure()
        fig.add_bar(x=miss.index, y=miss.missing, name="Missing bill", marker_color=ORANGE)
        fig.add_bar(x=miss.index, y=miss.zero, name="Zero bill", marker_color=GRAY)
        fig.update_layout(barmode="stack", title="Missing and zero bills per month", xaxis_title="Billing month",
                          yaxis_title="Households", height=380)
        b.plotly_chart(fig, width="stretch")
        st.markdown(f"""
**Cleaning applied**
- Bills such as `"1,159"` stored with thousands separators were parsed correctly (a naive conversion would delete the largest users).
- Missing bills stay missing — never treated as zero usage.
- 15 households billed 0 kWh in every month were removed as **invalid records** (occupied homes with a refrigerator according to the survey).
- Zero months and extreme values are kept; 18 strictly isolated one-month spikes are only flagged.
""")
        st.dataframe(obs.consumption.describe(percentiles=[.05, .25, .5, .75, .95, .99]).round(1).to_frame(f"Consumption ({U})").T,
                     width="stretch")
