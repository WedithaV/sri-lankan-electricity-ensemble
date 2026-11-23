"""Turn the 15-minute cumulative smart-meter registers into small, validated summaries.

Why: the raw 15-minute files hold ~45.6M rows (4.4 GB). EDA and the dashboard only need
aggregated patterns, so this script writes compact Parquet files to data/processed/.

How interval energy is derived (energy used in one 15-minute interval):
    1. Sort each household's readings by date+time (the raw rows are unordered and the
       raw ``timestamp`` column is rounded, so it is not used).
    2. Remove duplicate (household, time) readings.
    3. interval_kWh = importkwh(t) - importkwh(t - 15 min)
    4. The difference is VALID only when
         * the two readings are exactly 15 minutes apart (no gap is bridged), and
         * 0 <= interval_kWh <= 10 kWh  (a negative value means a meter reset/replacement;
           >10 kWh in 15 minutes = >40 kW average, beyond a household connection).
       Invalid or missing intervals are dropped -- they are NEVER treated as zero usage.
    5. Each interval is labelled by its START time (reading time - 15 minutes).

Outputs (data/processed/):
    smart_hourly_profile.parquet  household x day type x hour : mean kWh per hour
    smart_daily.parquet           household x date : kWh, only for complete days (96 valid intervals)
    smart_coverage.parquet        household : readings, valid intervals, first/last reading
    smart_interval_quality.json   counts of valid / invalid intervals by reason

Run from the project root:  python scripts/build_smart_meter_summaries.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import SMART_15MIN_DIR  # noqa: E402

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
STEP = pd.Timedelta(minutes=15)
MAX_INTERVAL_KWH = 10.0
CHUNK = 2_000_000


def read_compact(path: Path) -> pd.DataFrame:
    """Read only household, time and import register, converting each chunk to compact dtypes."""
    parts = []
    for ch in pd.read_csv(path, usecols=["household_ID", "date", "time", "importkwh(kwh)"],
                          dtype={"household_ID": "string", "date": "string", "time": "string"},
                          chunksize=CHUNK):
        parts.append(pd.DataFrame({
            "hh": ch["household_ID"].str[2:].astype("int32"),
            "ts": pd.to_datetime(ch["date"] + " " + ch["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"),
            "imp": pd.to_numeric(ch["importkwh(kwh)"], errors="coerce"),
        }))
    return pd.concat(parts, ignore_index=True)


def intervals_for_file(path: Path, quality: dict) -> pd.DataFrame:
    df = read_compact(path)
    quality["readings"] += len(df)
    df = df.dropna(subset=["ts", "imp"]).drop_duplicates(["hh", "ts"]).sort_values(["hh", "ts"])
    same_hh = df["hh"].eq(df["hh"].shift())
    d_ts = df["ts"] - df["ts"].shift()
    d_imp = df["imp"] - df["imp"].shift()
    pair = same_hh                                   # consecutive readings of one household
    exact = pair & d_ts.eq(STEP)
    neg = exact & (d_imp < 0)
    big = exact & (d_imp > MAX_INTERVAL_KWH)
    valid = exact & ~neg & ~big
    quality["consecutive_pairs"] += int(pair.sum())
    quality["gap_not_15min"] += int((pair & ~exact).sum())
    quality["negative_register_change"] += int(neg.sum())
    quality["over_10kwh_in_15min"] += int(big.sum())
    quality["valid_intervals"] += int(valid.sum())

    cov = df.groupby("hh").agg(readings=("ts", "size"), first=("ts", "min"), last=("ts", "max"))
    cov["valid_intervals"] = valid.groupby(df["hh"]).sum()
    out = pd.DataFrame({"hh": df.loc[valid, "hh"].to_numpy(),
                        "start": (df.loc[valid, "ts"] - STEP).to_numpy(),
                        "kwh": d_imp[valid].to_numpy()})
    return out, cov


def main() -> None:
    quality = dict(readings=0, consecutive_pairs=0, gap_not_15min=0, negative_register_change=0,
                   over_10kwh_in_15min=0, valid_intervals=0)
    hourly_parts, daily_parts, cov_parts = [], [], []
    for path in sorted(SMART_15MIN_DIR.glob("smart_15min_*.csv")):
        print("processing", path.name, flush=True)
        iv, cov = intervals_for_file(path, quality)
        iv["date"] = iv["start"].dt.normalize()
        iv["hour"] = iv["start"].dt.hour.astype("int8")
        # Sri Lankan weekend = Saturday and Sunday (public/Poya holidays not modelled).
        iv["day_type"] = np.where(iv["start"].dt.day_of_week >= 5, "Weekend", "Weekday")
        hourly_parts.append(iv.groupby(["hh", "day_type", "hour"])["kwh"].agg(["sum", "count"]).reset_index())
        daily_parts.append(iv.groupby(["hh", "date"])["kwh"].agg(["sum", "count"]).reset_index())
        cov_parts.append(cov.reset_index())
        del iv

    # Hourly profile: combine sums/counts across files, then mean kWh per hour (= 4 x mean interval kWh).
    h = pd.concat(hourly_parts).groupby(["hh", "day_type", "hour"])[["sum", "count"]].sum().reset_index()
    h["kwh_per_hour"] = 4 * h["sum"] / h["count"]
    h = h.rename(columns={"count": "n_intervals"}).drop(columns="sum")

    # Daily totals: a day crossing a file boundary is recombined; keep only complete days (96 intervals).
    d = pd.concat(daily_parts).groupby(["hh", "date"])[["sum", "count"]].sum().reset_index()
    quality["days_total"] = int(len(d))
    d = d[d["count"] == 96].rename(columns={"sum": "kwh"}).drop(columns="count")
    quality["complete_days"] = int(len(d))

    c = pd.concat(cov_parts).groupby("hh").agg(readings=("readings", "sum"), valid_intervals=("valid_intervals", "sum"),
                                               first=("first", "min"), last=("last", "max")).reset_index()
    for frame in (h, d, c):
        frame.insert(0, "household_ID", frame.pop("hh").map(lambda i: f"ID{i:04d}"))

    h.to_parquet(OUT / "smart_hourly_profile.parquet", index=False)
    d.to_parquet(OUT / "smart_daily.parquet", index=False)
    c.to_parquet(OUT / "smart_coverage.parquet", index=False)
    (OUT / "smart_interval_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
