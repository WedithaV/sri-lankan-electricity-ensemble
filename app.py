"""Sri Lankan Household Electricity Consumption Prediction Using Ensemble Learning (RECON-SL).

Streamlit dashboard. Run locally with:   streamlit run app.py
The app only LOADS saved results and the trained model -- it never trains a model.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="RECON-SL Ensemble Learning", page_icon="⚡", layout="wide",
                   initial_sidebar_state="auto")

from dashboard import page_compare, page_explore, page_overview, page_performance, page_predict  # noqa: E402
from dashboard.common import style  # noqa: E402

style()
pages = [
    st.Page(page_overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(page_explore.render, title="Dataset Exploration", icon="📊", url_path="explore"),
    st.Page(page_compare.render, title="Ensemble Model Comparison", icon="🏆", url_path="compare"),
    st.Page(page_performance.render, title="Model Performance Analysis", icon="🔍", url_path="performance"),
    st.Page(page_predict.render, title="Consumption Prediction", icon="⚡", url_path="predict"),]
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("### ⚡ RECON-SL")
    st.caption("Household electricity consumption prediction with Bagging, Boosting and Stacking.")
    st.caption("Data: LIRNEasia & LECO (CC-BY-4.0). All consumption values in **kWh**.")
nav.run()
