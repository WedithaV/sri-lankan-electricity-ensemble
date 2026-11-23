"""Cleaning of the monthly panel and household-level aggregation of Wave 1 survey data.

Cleaning rules (evidence in notebooks/02_exploratory_data_analysis.ipynb, section 2):

1. Missing readings (blank ``consumption``) stay missing. They are never filled with 0,
   because a missing bill is not zero usage.
2. MAIN EXPERIMENT: the 15 households whose consumption is exactly zero in every
   recorded month are removed as INVALID RECORDS. Verification (notebook 03, section 1):
   all 15 were occupied at Wave 1 (1-7 members, every one owns a refrigerator, 9-28
   appliances), and for the 7 with smart-meter data the linked meter also records
   0 kWh/day. An occupied home with a refrigerator cannot use zero electricity, so the
   linked meter is not the household's actual supply (meter-household linkage error).
   The 19 further households that never exceed LOW_ACTIVITY_MAX_KWH (5 kWh) are KEPT and
   only flagged (``is_low_activity_household``) for an additional robustness experiment
   that excludes all 34 (decision of 2026-11-23).
3. All other zero months are kept. Evidence: the month after an isolated zero is not
   inflated (median next/previous ratio = 1.0), so zeros are not "missed readings"
   that are caught up later; they behave like real zero-usage months.
4. Extreme values are kept. Most very high values belong to genuinely high-consuming
   households or to sustained level changes (moving in, new appliances).
   Only strictly isolated spikes -- more than 4x BOTH neighbouring months and more than
   200 kWh above them -- are *flagged* (``is_isolated_spike``) for a sensitivity
   analysis. The flag uses the following month, so it is a retrospective data-quality
   label and must never be used as a model feature.

Wave 1 aggregation: roster tables (one row per appliance/room/person) are turned into
one row per household so that a join with the monthly panel cannot duplicate rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_survey_table

LOW_ACTIVITY_MAX_KWH = 5.0  # never above this in any month -> flagged for the robustness experiment
SPIKE_RATIO = 4.0     # value must exceed this multiple of both neighbours
SPIKE_MIN_EXCESS = 200  # ... and be at least this many kWh above the larger neighbour


# --------------------------------------------------------------------------------------
# Monthly panel
# --------------------------------------------------------------------------------------
def flag_isolated_spikes(df: pd.DataFrame) -> pd.Series:
    """Boolean flag for strictly isolated one-month spikes (see module docstring).

    Computed only from the rows passed in, so callers can restrict it to a training
    period to avoid looking at test-period months.
    """
    df = df.sort_values(["household_ID", "month"])
    g = df.groupby("household_ID")["consumption"]
    prev, nxt = g.shift(1), g.shift(-1)
    larger = np.maximum(prev, nxt)
    flag = (
        prev.notna()
        & nxt.notna()
        & (df["consumption"] > SPIKE_RATIO * larger.clip(lower=1))
        & ((df["consumption"] - larger) > SPIKE_MIN_EXCESS)
    )
    return flag.reindex(df.index).fillna(False).astype(bool)


def clean_monthly(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules. Returns the cleaned frame and a log of what changed."""
    out = df.copy()
    hh_max = out.groupby("household_ID")["consumption"].transform("max")  # NaN-safe: ignores missing months
    # all-zero over recorded months (a household with no recorded month at all is also invalid)
    always_zero = hh_max.eq(0) | hh_max.isna()
    low_activity = hh_max.le(LOW_ACTIVITY_MAX_KWH) & ~always_zero
    log = {
        "input_rows": int(len(out)),
        "input_households": int(out["household_ID"].nunique()),
        "missing_readings_in_input": int(out["consumption"].isna().sum()),
        "always_zero_households_removed_invalid": int(out.loc[always_zero, "household_ID"].nunique()),
        "low_activity_households_kept_and_flagged": int(out.loc[low_activity, "household_ID"].nunique()),
    }
    out = out.loc[~always_zero].copy()
    out["is_low_activity_household"] = low_activity.loc[out.index]
    out["is_missing"] = out["consumption"].isna()
    out["is_zero"] = out["consumption"].eq(0)
    out["is_isolated_spike"] = flag_isolated_spikes(out)
    log.update(
        output_rows=int(len(out)),
        output_households=int(out["household_ID"].nunique()),
        missing_readings_kept_as_nan=int(out["is_missing"].sum()),
        zero_months_kept=int(out["is_zero"].sum()),
        isolated_spikes_flagged=int(out["is_isolated_spike"].sum()),
    )
    return out.reset_index(drop=True), log


# --------------------------------------------------------------------------------------
# Wave 1 household features
# --------------------------------------------------------------------------------------
# Appliance types (exact Wave 1 labels) that are large or frequent electricity users.
KEY_APPLIANCES = {
    "Refrigerator": "n_refrigerator",
    "Separate Freezer": "n_freezer",
    "Washing Machine": "n_washing_machine",
    "TV": "n_tv",
    "Electric Iron including electric steam iron": "n_iron",
    "Rice cooker": "n_rice_cooker",
    "Electric Kettle": "n_kettle",
    "Microwave": "n_microwave",
    "Electric Oven": "n_oven",
    "Electric cook tops (induction cookers, Infra-red cookers, hot plates)": "n_cooktop",
    "Geyser / Hot water systems for bathrooms which operate from electricity": "n_geyser",
    "Electric Water pump": "n_water_pump",
    "Computers": "n_computer",
    "Laptops": "n_laptop",
    "Clothes dryer": "n_clothes_dryer",
    "Electric vehicles (four wheelers) - cars,  vans, SUVs)": "n_ev_car",
}


def _yes(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower().eq("yes").astype(int)


def build_household_features_w1() -> pd.DataFrame:
    """One row per household with Wave 1 characteristics (structural, demographic, appliances).

    Skip-logic blanks in categorical survey answers are encoded as "Not applicable";
    remaining numeric gaps stay NaN and are imputed later inside the sklearn pipeline
    (fitted on training data only).
    """
    hi = load_survey_table(1, "household_information_and_history").set_index("household_ID")
    feats = pd.DataFrame(index=hi.index)

    # --- household & dwelling -------------------------------------------------------
    feats["household_size"] = hi["no_of_household_members"]
    feats["floor_area_sqft"] = hi["floor_area"]
    feats["no_of_electricity_meters"] = hi["no_of_electricity_meters"]
    feats["monthly_expenditure_lkr"] = hi["total_monthly_expenditure_of_last_month"]
    feats["has_home_business"] = _yes(hi["is_there_business_carried_out_in_the_household"])
    feats["has_renters_on_same_meter"] = hi["occupy_renters_boarders"].astype("string").str.startswith(
        ("Renters", "Boarders")).fillna(False).astype(int)
    cat_cols = {
        "type_of_house": "house_type",
        "built_year_of_the_house": "built_year",
        "socio_economic_class": "socio_economic_class",
        "electricity_provider_csc_area": "csc_area",
        "type_of_electricity_meter": "meter_type",
        "main_material_used_for_walls_of_the_house": "wall_material",
        "main_material_used_for_roof_of_the_house": "roof_material",
    }
    for src, dst in cat_cols.items():
        feats[dst] = hi[src].astype("string").str.strip().fillna("Not applicable")
    owns = hi["own_the_house_or_living_on_rent"].astype("string")
    # Order matters: the rent-free answer also contains the word "rent".
    feats["tenure"] = np.select(
        [owns.str.startswith("Yes"), owns.str.contains("without any payment", case=False),
         owns.str.contains("living on rent", case=False)],
        ["Owned", "Rent-free", "Rented"], "Other")

    # --- demographics ----------------------------------------------------------------
    dm = load_survey_table(1, "demographics")
    d = dm.groupby("household_ID").agg(
        n_children_under_15=("age", lambda a: int((a < 15).sum())),
        n_elderly_60_plus=("age", lambda a: int((a >= 60).sum())),
        mean_hours_at_home_per_week=("no_of_hours_stayed_at_home_during_last_week", "mean"),
    )
    d["n_employed"] = dm.assign(emp=dm["main_activity_engaged_in"].astype("string").str.startswith(
        "Engaged in economic activity").fillna(False)).groupby("household_ID")["emp"].sum()
    feats = feats.join(d)

    # --- appliances ------------------------------------------------------------------
    ap = load_survey_table(1, "appliances")
    counts = ap.groupby(["household_ID", "appliance_type"]).size().unstack(fill_value=0)
    for label, name in KEY_APPLIANCES.items():
        feats[name] = counts[label] if label in counts.columns else 0
    feats["n_appliances_total"] = counts.sum(axis=1)

    # --- air conditioners --------------------------------------------------------------
    ac = load_survey_table(1, "ac_roster")
    ac["weekly_hours"] = ac["no_of_hours_ac_was_on_during_daytime_last_week"] + ac[
        "no_of_hours_ac_was_on_during_night_last_week"]
    a = ac.groupby("household_ID").agg(n_ac=("ac_ID", "size"),
                                      ac_hours_per_week=("weekly_hours", "sum"))
    a["n_inverter_ac"] = ac.assign(inv=_yes(ac["is_the_ac_inverter_or_not"])).groupby("household_ID")["inv"].sum()
    feats = feats.join(a)

    # --- fans ------------------------------------------------------------------------
    fan = load_survey_table(1, "fan_roster")
    fan["weekly_hours"] = fan["no_of_hours_fan_was_on_during_daytime_last_week"] + fan[
        "no_of_hours_fan_was_on_during_night_last_week"]
    feats = feats.join(fan.groupby("household_ID").agg(n_fans=("fan_ID", "size"),
                                                       fan_hours_per_week=("weekly_hours", "sum")))

    # --- lighting --------------------------------------------------------------------
    li = load_survey_table(1, "light_roster")
    li["weekly_hours"] = li["no_of_hours_bulb_was_on_during_daytime_last_week"] + li[
        "no_of_hours_bulb_was_on_during_night_last_week"]
    li["is_led"] = li["type_of_the_bulb"].astype("string").str.contains("LED").fillna(False)
    feats = feats.join(li.groupby("household_ID").agg(n_bulbs=("light_ID", "size"),
                                                      bulb_hours_per_week=("weekly_hours", "sum"),
                                                      share_led_bulbs=("is_led", "mean")))

    # --- rooms -----------------------------------------------------------------------
    rm = load_survey_table(1, "room_roster")
    feats["n_rooms"] = rm.groupby("household_ID").size()
    feats["n_bedrooms"] = rm[rm["main_purpose_of_the_room"] == "Bedrooms"].groupby("household_ID").size()

    # --- generation, water heating, cooking --------------------------------------------
    g = load_survey_table(1, "electricity_generation_water_heating_cooking").set_index("household_ID")
    feats["has_solar"] = _yes(g["generate_electicity_using_solar_energy"])
    feats["has_backup_generator"] = _yes(g["have_backup_generator"])
    feats["has_battery_backup"] = _yes(g["have_system_to_store_backup_energy"])
    feats["cooks_with_grid_electricity"] = _yes(g["electricity_from_national_grid_used_for_cooking"])
    heat = g["water_heating_method_for_bathing"].astype("string")
    feats["water_heating"] = np.select(
        [heat.str.startswith("None"), heat.str.contains("solely by the grid"),
         heat.str.contains("both by solar and the grid"), heat.str.contains("solely by solar"),
         heat.str.contains("electric kettle")],
        ["None", "Grid electric", "Solar + grid", "Solar", "Kettle/heater"], "Other non-electric")

    # Households absent from a roster own zero of those items (not unknown).
    zero_if_absent = ["n_ac", "ac_hours_per_week", "n_inverter_ac", "n_fans", "fan_hours_per_week",
                      "n_bulbs", "bulb_hours_per_week", "n_bedrooms", "n_children_under_15",
                      "n_elderly_60_plus", "n_employed", "n_appliances_total"] + list(KEY_APPLIANCES.values())
    feats[zero_if_absent] = feats[zero_if_absent].fillna(0)
    feats.index.name = "household_ID"
    return feats.reset_index()
