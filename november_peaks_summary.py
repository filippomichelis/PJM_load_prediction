#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from make_datasets import load_load_history, TZ_NAME


def build_november_peak_summary(load_dir: Path, out_path: Path, years_back: int = 3) -> pd.DataFrame:
    load_df = load_load_history(load_dir, TZ_NAME)
    cutoff_start = (
        pd.Timestamp.now(tz=TZ_NAME).normalize()
        - pd.DateOffset(years=years_back)
    )
    load_df = load_df[load_df["timestamp_local"] >= cutoff_start]
    november = load_df[load_df["timestamp_local"].dt.month == 11].copy()

    records = []
    for area, area_df in november.groupby("load_area", sort=False):
        if area_df.empty:
            continue
        min_row = area_df.loc[area_df["mw"].idxmin()]
        max_row = area_df.loc[area_df["mw"].idxmax()]
        records.append(
            {
                "load_area": area,
                "min_mw": float(min_row["mw"]),
                "min_timestamp": min_row["timestamp_local"],
                "max_mw": float(max_row["mw"]),
                "max_timestamp": max_row["timestamp_local"],
            }
        )

    summary_df = pd.DataFrame(records).sort_values("load_area").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(summary_df, out_path)
    return summary_df


if __name__ == "__main__":
    output_path = Path("artifacts/november_peak_summary.joblib")
    summary = build_november_peak_summary(Path("data"), output_path)
    print(f"Saved November peak summary for {len(summary)} areas to {output_path.resolve()}")
