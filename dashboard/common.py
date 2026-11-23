"""Shared loaders, styling and small UI components for the Streamlit dashboard.

Everything is read from saved files (artifacts/, models/, data/processed/). No model is
trained while the dashboard runs. All paths are relative to the project root, so the app
works locally on Windows and on Streamlit Community Cloud (Linux).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART, MODELS, PROCESSED = ROOT / "artifacts", ROOT / "models", ROOT / "data" / "processed"

NAVY, TEAL, BLUE, ORANGE = "#1B2A4A", "#0F9D8F", "#2A78D6", "#EB6834"
GRAY, LIGHT, INK2, GRID = "#9AA5B1", "#F3F5F7", "#52514E", "#E3E6EA"
FAMILY_COLORS = {"Baseline": GRAY, "Bagging": TEAL, "Boosting": BLUE, "Ensemble combination": ORANGE}

# Plotly template: thin marks, recessive grid, navy text, consistent model-family colours.
pio.templates["recon"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", color=NAVY, size=13),
    title=dict(font=dict(size=16, color=NAVY), x=0, xanchor="left"),
    colorway=[TEAL, BLUE, ORANGE], paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor=GRID, linecolor=GRAY, zeroline=False, ticks="outside", tickcolor=GRAY),
    yaxis=dict(gridcolor=GRID, linecolor=GRAY, zeroline=False, ticks="outside", tickcolor=GRAY),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title_text=""),
    margin=dict(l=10, r=10, t=75, b=10), hoverlabel=dict(bgcolor="white", font_color=NAVY)))
pio.templates.default = "recon"

CSS = f"""
<style>
.block-container {{padding-top: 2rem; max-width: 1250px;}}
h1, h2, h3 {{color: {NAVY}; letter-spacing: -0.01em;}}
.kpi {{background: white; border: 1px solid {GRID}; border-radius: 14px; padding: 14px 16px; height: 100%;
       box-shadow: 0 1px 2px rgba(27,42,74,0.05);}}
.kpi .label {{font-size: 0.78rem; color: {INK2}; text-transform: uppercase; letter-spacing: .04em;}}
.kpi .value {{font-size: 1.55rem; font-weight: 700; color: {NAVY}; line-height: 1.25; margin-top: 2px;}}
.kpi .sub {{font-size: 0.8rem; color: {INK2}; margin-top: 2px;}}
.kpi.accent {{border-top: 4px solid {TEAL};}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #24406b 60%, {TEAL} 140%); color: white; border-radius: 18px;
        padding: 26px 28px; margin-bottom: 18px;}}
.hero h1 {{color: white; margin: 0 0 6px 0; font-size: 1.9rem;}}
.hero p {{color: #E6EEF6; margin: 0; font-size: 1.02rem;}}
.card {{background: {LIGHT}; border-radius: 14px; padding: 16px 18px; height: 100%;}}
.card h4 {{margin: 0 0 6px 0; color: {NAVY};}}
.pill {{display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: #E2F4F2; color: {TEAL};}}
.result {{background: white; border: 2px solid {TEAL}; border-radius: 18px; padding: 22px; text-align: center;}}
.result .big {{font-size: 2.6rem; font-weight: 800; color: {NAVY};}}
.note {{font-size: .85rem; color: {INK2};}}
@media (max-width: 640px) {{ .hero h1 {{font-size: 1.4rem;}} .kpi .value {{font-size: 1.25rem;}} }}
</style>
"""


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: bool = False):
    st.markdown(f'<div class="kpi{" accent" if accent else ""}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><div>{body}</div></div>', unsafe_allow_html=True)


def missing_artifacts_message(err: Exception):
    st.error("Some saved results could not be loaded. Run the training scripts first "
             "(see README → *Reproduce the results*).")
    st.caption(f"Details: {err}")


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads((ART / "model_metadata.json").read_text())


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    return pd.read_csv(ART / "model_results.csv")


@st.cache_data(show_spinner=False)
def load_cv() -> pd.DataFrame:
    return pd.read_csv(ART / "cross_validation_results.csv")


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(ART / "test_predictions.csv", parse_dates=["month"])


@st.cache_data(show_spinner=False)
def load_importance() -> pd.DataFrame:
    return pd.read_csv(ART / "feature_importance.csv")


@st.cache_data(show_spinner=False)
def load_monthly() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "monthly_clean.parquet")


@st.cache_data(show_spinner=False)
def load_household_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "household_features_w1.parquet")


@st.cache_data(show_spinner=False)
def load_hourly_profile() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "smart_hourly_profile.parquet")


@st.cache_resource(show_spinner=False)
def load_bundle(name: str = "best_model.joblib"):
    from src.predict import load_bundle as _load
    return _load(MODELS / name)


LABELS = {
    "naive_last_month": "Naive: last month", "naive_mean_3m": "Naive: 3-month mean", "dummy_mean": "Dummy (mean)",
    "linear": "Linear Regression", "ridge": "Ridge Regression", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGradientBoosting", "voting": "Voting", "stacking": "Stacking",
}


def pretty(label: str) -> str:
    return LABELS.get(label, label.replace("_", " ").title())
