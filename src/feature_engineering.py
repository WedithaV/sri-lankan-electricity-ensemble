"""Leakage-safe feature engineering for one-month-ahead consumption prediction.

PREDICTION SETTING
------------------
For each household and billing month t we predict ``consumption`` (kWh) of month t.
The prediction is made at the START of billing month t ("prediction date"). At that
moment the following information is available:

* the household's bills for billing months <= t-1 (never month t or later);
* Wave 1 survey answers ONLY if the household's Wave 1 interview took place strictly
  before the first day of month t (actual dates from survey_dates.csv). Otherwise every
  survey feature is unavailable (NaN) and ``survey_available`` = 0.
* Waves 2 and 3 are never used (collected after the consumption period).

Rolling one-month-ahead evaluation: in the test period (Jun-Oct 2024) the feature row for
July uses June's ACTUAL bill. That is legitimate because June's bill is known before
July's prediction is made -- and June's own prediction was already evaluated. Because
every feature below is built from ``shift(k >= 1)`` of the actual series, this rule holds
automatically; tests/test_feature_leakage.py verifies it by changing future values and
checking that no earlier feature changes.

Every feature is explained in FEATURE_DESCRIPTIONS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ feature groups
HISTORY_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_median_3", "roll_std_6",
    "hist_mean", "hist_max", "change_1",
    "zero_count_6", "missing_count_6",
]
CALENDAR_FEATURES = ["month_of_year", "month_sin", "month_cos"]

SURVEY_NUMERIC = [
    "household_size", "floor_area_sqft", "monthly_expenditure_lkr", "n_rooms", "n_bedrooms",
    "n_children_under_15", "n_elderly_60_plus", "n_employed", "mean_hours_at_home_per_week",
    "n_ac", "n_inverter_ac", "ac_hours_per_week", "n_fans", "fan_hours_per_week", "n_bulbs",
    "share_led_bulbs", "n_refrigerator", "n_washing_machine", "n_geyser", "n_water_pump",
    "n_cooktop", "n_oven", "n_microwave", "n_kettle", "n_computer", "n_laptop", "n_appliances_total",
    "has_solar", "has_home_business", "cooks_with_grid_electricity", "no_of_electricity_meters",
]
SURVEY_CATEGORICAL = ["socio_economic_class", "house_type", "built_year", "csc_area", "tenure",
                      "water_heating", "meter_type"]
SURVEY_FLAG = ["survey_available"]

# Columns that must NEVER be model inputs (identifiers, the target, current-month or
# retrospective data-quality flags).
FORBIDDEN_FEATURES = {"household_ID", "month", "consumption", "is_missing", "is_zero",
                      "is_isolated_spike", "is_low_activity_household", "wave1_date", "target"}

FEATURE_SETS = {
    "history_only": {"numeric": HISTORY_FEATURES + CALENDAR_FEATURES, "categorical": []},
    "history_plus_survey": {"numeric": HISTORY_FEATURES + CALENDAR_FEATURES + SURVEY_FLAG + SURVEY_NUMERIC,
                            "categorical": SURVEY_CATEGORICAL},
    # POST-HOC DIAGNOSTIC ONLY (added after the test results showed a large October 2024 bias):
    # are calendar-month features / last year's bill responsible? Never used for model selection.
    "diag_no_calendar": {"numeric": HISTORY_FEATURES, "categorical": []},
    "diag_no_calendar_no_lag12": {"numeric": [f for f in HISTORY_FEATURES if f != "lag_12"], "categorical": []},
}

FEATURE_DESCRIPTIONS = {
    "lag_1": "Bill of the previous month (t-1), kWh",
    "lag_2": "Bill two months before (t-2), kWh",
    "lag_3": "Bill three months before (t-3), kWh",
    "lag_6": "Bill six months before (t-6), kWh",
    "lag_12": "Bill of the same billing month last year (t-12), kWh — captures seasonality",
    "roll_mean_3": "Mean of the bills t-3..t-1 (observed months only), kWh",
    "roll_mean_6": "Mean of the bills t-6..t-1, kWh",
    "roll_median_3": "Median of the bills t-3..t-1 — robust to a single spike, kWh",
    "roll_std_6": "Standard deviation of the bills t-6..t-1 — how stable the household is, kWh",
    "hist_mean": "Mean of ALL earlier bills (expanding, up to t-1), kWh — long-run level",
    "hist_max": "Highest earlier bill (up to t-1), kWh",
    "change_1": "lag_1 - lag_2: most recent month-to-month change, kWh",
    "zero_count_6": "Number of zero bills among t-6..t-1",
    "missing_count_6": "Number of missing bills among t-6..t-1",
    "month_of_year": "Calendar month of the billing month (1-12)",
    "month_sin": "sin(2*pi*month/12) — cyclic month encoding (Dec is next to Jan)",
    "month_cos": "cos(2*pi*month/12) — cyclic month encoding",
    "survey_available": "1 if the Wave 1 interview happened before the prediction date, else 0",
}


# ------------------------------------------------------------------ history features
def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling features computed ONLY from earlier months of the same household.

    ``panel`` must contain every household-month (missing readings as NaN), so that a
    shift by k rows is exactly a shift by k months.
    """
    df = panel.sort_values(["household_ID", "month"]).reset_index(drop=True)
    # Safety: consecutive rows of a household must be consecutive months.
    step = df.groupby("household_ID")["month"].diff().dropna()
    if not step.dt.days.between(28, 31).all():
        raise ValueError("Panel has gaps or duplicates in months; shift(k) would not equal k months.")

    g = df.groupby("household_ID")["consumption"]
    for k in [1, 2, 3, 6, 12]:
        df[f"lag_{k}"] = g.shift(k)

    past = g.shift(1)                      # value of t-1 placed on row t: everything below uses only <= t-1
    gp = past.groupby(df["household_ID"])
    df["roll_mean_3"] = gp.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["roll_mean_6"] = gp.transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["roll_median_3"] = gp.transform(lambda s: s.rolling(3, min_periods=1).median())
    df["roll_std_6"] = gp.transform(lambda s: s.rolling(6, min_periods=2).std())
    df["hist_mean"] = gp.transform(lambda s: s.expanding(min_periods=1).mean())
    df["hist_max"] = gp.transform(lambda s: s.expanding(min_periods=1).max())
    df["change_1"] = df["lag_1"] - df["lag_2"]
    df["zero_count_6"] = gp.transform(lambda s: s.eq(0).astype(float).rolling(6, min_periods=1).sum())
    # missing_count_6: positions t-6..t-1 that exist in the panel but have no reading
    is_na = past.isna().astype(float).where(g.cumcount() > 0)  # row 0 has no past at all -> NaN
    df["missing_count_6"] = is_na.groupby(df["household_ID"]).transform(lambda s: s.rolling(6, min_periods=1).sum())
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mo = df["month"].dt.month
    df["month_of_year"] = mo
    df["month_sin"] = np.sin(2 * np.pi * mo / 12)
    df["month_cos"] = np.cos(2 * np.pi * mo / 12)
    return df


# ------------------------------------------------------------------ survey availability
def add_survey_features(df: pd.DataFrame, household_features: pd.DataFrame,
                        survey_dates: pd.DataFrame) -> pd.DataFrame:
    """Join Wave 1 features, masking them for months whose prediction date precedes the interview.

    Prediction date of billing month t = first day of month t. Survey information is
    available only if wave1 interview date < prediction date.
    """
    hf = household_features.set_index("household_ID")[SURVEY_NUMERIC + SURVEY_CATEGORICAL]
    out = df.join(hf, on="household_ID")
    w1 = survey_dates.set_index("household_ID")["wave1"].rename("wave1_date")
    out = out.join(w1, on="household_ID")
    available = out["wave1_date"].notna() & (out["wave1_date"] < out["month"])
    out["survey_available"] = available.astype(int)
    out.loc[~available, SURVEY_NUMERIC] = np.nan
    out[SURVEY_CATEGORICAL] = out[SURVEY_CATEGORICAL].astype("object")
    out.loc[~available, SURVEY_CATEGORICAL] = np.nan
    return out


def build_feature_table(monthly_clean: pd.DataFrame, household_features: pd.DataFrame,
                        survey_dates: pd.DataFrame) -> pd.DataFrame:
    """Full household-month feature table (all months; target may be NaN)."""
    df = add_history_features(monthly_clean)
    df = add_calendar_features(df)
    df = add_survey_features(df, household_features, survey_dates)
    return df
