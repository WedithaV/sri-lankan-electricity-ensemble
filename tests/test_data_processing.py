"""Tests for the data-loading / parsing rules."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import MONTHLY_PATH, load_monthly_consumption, parse_consumption  # noqa: E402


def test_parse_consumption_handles_thousands_separator_and_blanks():
    s = pd.Series(["96", "1,159", "", "0", " 2,629 "])
    out = parse_consumption(s)
    assert out.iloc[0] == 96
    assert out.iloc[1] == 1159          # comma removed, NOT turned into NaN
    assert np.isnan(out.iloc[2])        # blank stays missing, NOT zero
    assert out.iloc[3] == 0
    assert out.iloc[4] == 2629


def test_parse_consumption_rejects_unknown_formats():
    with pytest.raises(ValueError):
        parse_consumption(pd.Series(["12", "abc"]))


# ---------------------------------------------------------------- cleaning rules
from src.data_processing import build_household_features_w1, clean_monthly, flag_isolated_spikes  # noqa: E402


def _panel(values_by_household):
    """Build a small synthetic monthly panel: {household: [v1, v2, ...]}."""
    months = pd.date_range("2023-01-01", periods=len(next(iter(values_by_household.values()))), freq="MS")
    rows = [(h, mo, v) for h, vals in values_by_household.items() for mo, v in zip(months, vals)]
    return pd.DataFrame(rows, columns=["household_ID", "month", "consumption"])


def test_isolated_spike_flagged_but_sustained_level_change_is_not():
    df = _panel({
        "SPIKE": [100, 100, 900, 100, 100],   # isolated: >4x both neighbours and >200 kWh above
        "STEP":  [100, 100, 900, 900, 900],   # sustained change: genuine, must NOT be flagged
        "HIGH":  [2000, 2100, 2500, 2050, 1990],  # genuinely high user
    })
    flags = flag_isolated_spikes(df)
    flagged = df.loc[flags, ["household_ID", "month"]]
    assert list(flagged.household_ID) == ["SPIKE"]
    assert flagged.month.iloc[0] == pd.Timestamp("2023-03-01")


def test_spike_at_series_edge_is_not_flagged():
    # The first/last month has only one neighbour, so it cannot be shown to be isolated.
    df = _panel({"EDGE": [900, 100, 100, 100, 100]})
    assert not flag_isolated_spikes(df).any()


def test_clean_monthly_removes_only_always_zero_and_flags_low_activity():
    df = _panel({
        "ALWAYSZERO": [0, 0, np.nan, 0, 0],   # zero in every recorded month -> removed (invalid record)
        "LOW":        [0, 1, np.nan, 0, 2],   # never above 5 kWh -> KEPT, flagged for robustness experiment
        "ACTIVE":     [80, 0, np.nan, 90, 85],  # zero and missing months must be kept as they are
    })
    out, log = clean_monthly(df)
    assert set(out.household_ID) == {"LOW", "ACTIVE"}
    assert log["always_zero_households_removed_invalid"] == 1
    assert log["low_activity_households_kept_and_flagged"] == 1
    assert out.loc[out.household_ID == "LOW", "is_low_activity_household"].all()
    assert not out.loc[out.household_ID == "ACTIVE", "is_low_activity_household"].any()
    active = out[out.household_ID == "ACTIVE"].set_index("month").consumption
    assert active.iloc[1] == 0            # zero kept (not treated as missing)
    assert np.isnan(active.iloc[2])       # missing kept as NaN (not filled with 0)
    act = out[out.household_ID == "ACTIVE"]
    assert act.is_missing.sum() == 1 and act.is_zero.sum() == 1


@pytest.mark.skipif(not MONTHLY_PATH.exists(), reason="raw RECON-SL data not downloaded")
def test_household_features_one_row_per_household():
    f = build_household_features_w1()
    assert f.household_ID.is_unique and len(f) == 4063
    assert set(f.tenure.unique()) <= {"Owned", "Rented", "Rent-free", "Other"}
    # roster counts: absent from roster means zero owned, never missing
    for col in ["n_ac", "n_fans", "n_refrigerator", "n_bulbs"]:
        assert f[col].notna().all() and (f[col] >= 0).all()


@pytest.mark.skipif(not MONTHLY_PATH.exists(), reason="raw RECON-SL data not downloaded")
def test_monthly_panel_matches_audit():
    df = load_monthly_consumption()
    assert df.shape == (101_575, 3)
    assert df["household_ID"].nunique() == 4063
    assert df["month"].nunique() == 25
    assert df["consumption"].isna().sum() == 590       # only genuine blanks are missing
    assert not df.duplicated(["household_ID", "month"]).any()
    assert (df["consumption"].dropna() >= 0).all()
