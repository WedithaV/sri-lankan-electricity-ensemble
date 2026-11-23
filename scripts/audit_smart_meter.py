"""Audit of the RECON-SL smart-meter files (15-minute and 6-hour).

The raw files are large (~5.7 GB uncompressed, ~50M rows), so they are read in
chunks with only the columns needed for each check. Results are written to
reports/smart_meter_audit.json and reports/unit_check_6hour_vs_monthly.csv so
that notebook 01 can display them without re-reading the raw files.

Run from the project root:
    python scripts/audit_smart_meter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import SMART_15MIN_DIR, SMART_6HOUR_DIR, load_monthly_consumption  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
CHUNK = 2_000_000


def audit_file(path: Path, id_col: str, date_col: str, time_col: str, import_col: str) -> tuple[dict, pd.DataFrame]:
    """Chunked pass: row counts, missing values, date range. Returns summary + slim frame for ordering checks."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    n_rows, missing = 0, pd.Series(0, index=header, dtype="int64")
    slim_parts = []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        n_rows += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0)
        slim = chunk[[id_col, date_col, time_col, import_col]].copy()
        slim.columns = ["hh", "date", "time", "imp"]
        slim["imp"] = pd.to_numeric(slim["imp"], errors="coerce")  # some 6-hour values are text
        # IDs look like "ID0123": store the number as int32 to save memory on ~20M-row files.
        bad_ids = ~slim["hh"].astype(str).str.fullmatch(r"ID\d+")
        if bad_ids.any():
            raise ValueError(f"Unexpected household_ID format in {path.name}: {slim.loc[bad_ids, 'hh'].unique()[:5]}")
        slim["hh"] = slim["hh"].str[2:].astype("int32")
        slim["ts"] = pd.to_datetime(slim["date"] + " " + slim["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        slim_parts.append(slim.drop(columns=["date", "time"]))
    slim = pd.concat(slim_parts, ignore_index=True)

    summary = {
        "file": path.name,
        "rows": int(n_rows),
        "columns": header,
        "missing_per_column": {k: int(v) for k, v in missing.items() if v > 0},
        "unique_households": int(slim["hh"].nunique()),
        "invalid_timestamps": int(slim["ts"].isna().sum()),
        "date_min": str(slim["ts"].min()),
        "date_max": str(slim["ts"].max()),
        "import_non_numeric_or_missing": int(slim["imp"].isna().sum()),
        "import_negative_values": int((slim["imp"] < 0).sum()),
        "duplicate_household_timestamp": int(slim.duplicated(["hh", "ts"]).sum()),
        "rows_out_of_time_order_in_file": int((slim.groupby("hh")["ts"].diff() < pd.Timedelta(0)).sum()),
    }
    return summary, slim


def register_checks(slim: pd.DataFrame, expected_step: pd.Timedelta) -> dict:
    """Sort by household/time and check the cumulative import register."""
    s = slim.dropna(subset=["ts"]).drop_duplicates(["hh", "ts"]).sort_values(["hh", "ts"])
    d_imp = s.groupby("hh")["imp"].diff()
    d_ts = s.groupby("hh")["ts"].diff()
    out = {
        "register_decreases (possible reset/meter change)": int((d_imp < 0).sum()),
        "households_with_any_decrease": int(s.loc[d_imp < 0, "hh"].nunique()),
        "median_gap_between_readings": str(d_ts.median()),
        f"gaps_longer_than_{expected_step}": int((d_ts > expected_step * 1.5).sum()),
        "gaps_longer_than_1_day": int((d_ts > pd.Timedelta(days=1)).sum()),
        "interval_kwh_quantiles (diff of register)": {
            str(q): float(d_imp.quantile(q)) for q in [0.01, 0.5, 0.99, 0.9999]
        },
    }
    return out


def main() -> None:
    report: dict = {"15min": [], "6hour": []}
    all_ids: dict[str, set] = {"15min": set(), "6hour": set()}

    six_slims = []
    for path in sorted(SMART_6HOUR_DIR.glob("*.csv")):
        print("auditing", path.name, flush=True)
        summ, slim = audit_file(path, "household_ID", "DATE", "TIME", "TOTAL_IMPORT (kWh)")
        summ.update(register_checks(slim, pd.Timedelta(hours=6)))
        report["6hour"].append(summ)
        all_ids["6hour"] |= set(int(x) for x in slim["hh"].unique())
        six_slims.append(slim)

    for path in sorted(SMART_15MIN_DIR.glob("*.csv")):
        print("auditing", path.name, flush=True)
        summ, slim = audit_file(path, "household_ID", "date", "time", "importkwh(kwh)")
        summ.update(register_checks(slim, pd.Timedelta(minutes=15)))
        report["15min"].append(summ)
        all_ids["15min"] |= set(int(x) for x in slim["hh"].unique())
        del slim

    monthly = load_monthly_consumption()
    monthly_ids = set(monthly["household_ID"].str[2:].astype(int).unique())
    report["household_overlap"] = {
        "monthly_households": len(monthly_ids),
        "6hour_households": len(all_ids["6hour"]),
        "15min_households": len(all_ids["15min"]),
        "6hour_in_monthly": len(all_ids["6hour"] & monthly_ids),
        "15min_in_monthly": len(all_ids["15min"] & monthly_ids),
        "6hour_and_15min": len(all_ids["6hour"] & all_ids["15min"]),
        "smart_ids_not_in_monthly_examples": [f"ID{i:04d}" for i in sorted((all_ids["6hour"] | all_ids["15min"]) - monthly_ids)[:10]],
    }

    # ---- Unit check: calendar-month import from the 6-hour register vs billed monthly consumption ----
    # Calendar-month kWh = register at first reading of month m+1 minus register at first reading of month m.
    six = pd.concat(six_slims, ignore_index=True).dropna(subset=["ts", "imp"]).sort_values(["hh", "ts"])
    six["month"] = six["ts"].dt.to_period("M").dt.to_timestamp()
    first = six.groupby(["hh", "month"], as_index=False).first()[["hh", "month", "imp"]]
    first["next_imp"] = first.groupby("hh")["imp"].shift(-1)
    first["next_month"] = first.groupby("hh")["month"].shift(-1)
    first = first[first["next_month"] == first["month"] + pd.offsets.MonthBegin(1)]
    first["smart_kwh"] = first["next_imp"] - first["imp"]
    monthly["hh"] = monthly["household_ID"].str[2:].astype("int32")
    merged = first.merge(monthly, on=["hh", "month"], how="inner")
    merged = merged.dropna(subset=["consumption", "smart_kwh"])
    merged = merged[(merged["smart_kwh"] >= 0)]
    merged[["household_ID", "month", "smart_kwh", "consumption"]].to_csv(REPORTS / "unit_check_6hour_vs_monthly.csv", index=False)

    # Meter replacements/resets create impossible register jumps (up to ~200,000 kWh in one month).
    # They are excluded from the comparison; the count is reported.
    n_jumps = int((merged["smart_kwh"] > 5000).sum())
    merged = merged[merged["smart_kwh"] <= 5000]
    valid = merged[merged["consumption"] > 0]
    ratio = valid["consumption"] / valid["smart_kwh"].replace(0, np.nan)
    report["unit_check"] = {
        "excluded_register_jumps_over_5000_kwh": n_jumps,
        "matched_household_months": int(len(merged)),
        "matched_households": int(merged["hh"].nunique()),
        "mean_billed": float(merged["consumption"].mean()),
        "mean_smart_kwh": float(merged["smart_kwh"].mean()),
        "pearson_corr_billed_vs_smart_kwh": float(merged["consumption"].corr(merged["smart_kwh"])),
        "spearman_corr_billed_vs_smart_kwh": float(merged["consumption"].corr(merged["smart_kwh"], method="spearman")),
        "median_ratio_billed_over_smart_kwh": float(ratio.median()),
        "ratio_iqr": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))],
        "share_within_10pct": float(((ratio > 0.9) & (ratio < 1.1)).mean()),
        # Billing cycles may not align to calendar months, so also compare with neighbouring months.
        "corr_with_smart_kwh_previous_month": float(
            merged.assign(prev=merged.groupby("hh")["smart_kwh"].shift(1))[["consumption", "prev"]].corr().iloc[0, 1]
        ),
        "corr_with_smart_kwh_next_month": float(
            merged.assign(nxt=merged.groupby("hh")["smart_kwh"].shift(-1))[["consumption", "nxt"]].corr().iloc[0, 1]
        ),
    }

    with open(REPORTS / "smart_meter_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
