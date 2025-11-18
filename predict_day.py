import argparse
import datetime
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from projection_stacker import apply_projection_correction
from projection_stacker import train_projection_for_window

warnings.filterwarnings("ignore")

# Macros / constants

parser = argparse.ArgumentParser(description="Generate per-load-area predictions.")
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print intermediate logs (final submission-style output always prints).",
)
cli_args = parser.parse_args()
VERBOSE = cli_args.verbose

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

today = datetime.datetime.now().date()
TARGET_DAY = today + datetime.timedelta(days=1)
TZ = "America/New_York"
MODEL_DIR = Path("artifacts")
FULL_DS = Path("data/full_datasets.joblib")
REGION_MODELS = joblib.load(MODEL_DIR / "region_models.joblib")
DATASETS = joblib.load(FULL_DS)                 
LOAD_HISTORY_CSV = Path("data/hrl_load_metered_2025.csv")
NOVEMBER_PEAK_SUMMARY = Path("artifacts/november_peak_summary.joblib")

AREAS = [
"AECO","AEPAPT","AEPIMP","AEPKPT","AEPOPT","AP","BC","CE","DAY","DEOK",
"DOM","DPLCO","DUQ","EASTON","EKPC","JC","ME","OE","OVEC","PAPWR","PE",
"PEPCO","PLCO","PN","PS","RECO",
"SMECO", "UGI", "VMEU"
]

# Load models and datasets
REGION_MODELS = joblib.load(MODEL_DIR / "region_models.joblib")
DATASETS = joblib.load(FULL_DS)

def load_peak_ranges(summary_path: Path) -> dict[str, tuple[float, float]]:
    if not summary_path.exists():
        return {}
    summary_df = joblib.load(summary_path)
    ranges = {}
    for _, row in summary_df.iterrows():
        ranges[str(row["load_area"])] = (float(row["min_mw"]), float(row["max_mw"]))
    return ranges

PEAK_RANGES = load_peak_ranges(NOVEMBER_PEAK_SUMMARY)

last_measurements = {}
for area, (df, _) in DATASETS.items():
    area_df = df.dropna(subset=["mw"])
    if area_df.empty:
        last_measurements[area] = None
        continue
    last_ts = area_df["timestamp_local"].max()
    last_measurements[area] = last_ts

# Day (and datasets) dependent macros
ts = last_measurements['AECO']
day_str = ts.strftime("%Y-%m-%d")
four_days_before_ts = (ts.normalize() - pd.Timedelta(days=4)).strftime("%Y-%m-%d %H:%M:%S")
day_str_extended = ts.normalize().strftime("%Y-%m-%d %H:%M:%S")

STARTING_DAY_FOR_EXTRAPOLATION = day_str
vprint(f"Starting day for extrapolation: {STARTING_DAY_FOR_EXTRAPOLATION}")

summary = train_projection_for_window(
    window_start=four_days_before_ts,
    window_end=day_str_extended,
    model_dir=Path("artifacts"),
    full_datasets_path=Path("data/full_datasets.joblib"),
)
vprint("Trained projection summary window:", four_days_before_ts, "to", day_str_extended)

# load raw 2025 history once for fallback areas
load_history = (
    pd.read_csv(
        LOAD_HISTORY_CSV,
        usecols=["datetime_beginning_ept", "load_area", "mw"],
        parse_dates=["datetime_beginning_ept"],
    )
    .rename(columns={"datetime_beginning_ept": "timestamp_local"})
)
load_history["timestamp_local"] = load_history["timestamp_local"].dt.tz_localize(
    TZ, nonexistent="shift_forward", ambiguous="NaT"
)
load_history = load_history.dropna(subset=["timestamp_local"])

fallback_lookup = {
    area: df.reset_index(drop=True)
    for area, df in load_history.groupby("load_area", sort=False)
}

day_start = pd.Timestamp(TARGET_DAY, tz=TZ)
day_end = day_start + pd.Timedelta(days=1)
mask = lambda df: (df["timestamp_local"] >= day_start) & (df["timestamp_local"] < day_end)

def average_last_three_days(area_df: pd.DataFrame) -> pd.DataFrame:
    window = area_df[
        (area_df["timestamp_local"] >= day_start - pd.Timedelta(days=3))
        & (area_df["timestamp_local"] < day_start)
    ]
    if window.empty:
        return pd.DataFrame()

    window = window.sort_values("timestamp_local").copy()
    window["hour"] = window["timestamp_local"].dt.hour
    hourly = window.groupby("hour")["mw"].mean()
    fallback = window["mw"].mean()

    target_hours = pd.date_range(day_start, periods=24, freq="H", tz=TZ)
    values = [hourly.get(h, fallback) for h in range(24)]
    return pd.DataFrame(
        {
            "load_area": area_df["load_area"].iloc[0],
            "timestamp_local": target_hours,
            "mw_pred": values,
        }
    )
# list of days between STARTING_DAY_FOR_EXTRAPOLATION and TARGET_DAY
days_to_predict = pd.date_range(
    start=STARTING_DAY_FOR_EXTRAPOLATION,
    end=TARGET_DAY,
    freq="D",
    tz=TZ,
)

# produce a prediction for the starting day
for day in days_to_predict:
    day_start = pd.Timestamp(day)  # already tz-aware
    day_end = day_start + pd.Timedelta(days=1)
    day_mask = lambda df, start=day_start, end=day_end: (
        (df["timestamp_local"] >= start) & (df["timestamp_local"] < end)
    )

    pred_rows = []
    for area in AREAS:
        if area in DATASETS:
            area_df, feature_cols = DATASETS[area]
        elif area in fallback_lookup:
            area_df = fallback_lookup[area]
            feature_cols = None
        else:
            vprint(f"Warning: no data for {area}, skipping.")
            continue

        if area not in REGION_MODELS:
            avg_df = average_last_three_days(area_df)
            if avg_df.empty:
                vprint(f"Warning: insufficient history for {area} fallback")
                continue
            pred_rows.append(avg_df)
            continue

        payload = REGION_MODELS[area]
        day_df = area_df.loc[day_mask(area_df)].copy()
        if day_df.empty:
            vprint(f"Warning: no rows for {area} on {day.date()}")
            continue

        X = day_df[payload["kept_features"]]
        preds = payload["pipeline"].predict(X)
        pred_rows.append(
            pd.DataFrame(
                {
                    "load_area": area,
                    "timestamp_local": day_df["timestamp_local"],
                    "mw_pred": preds,
                }
            )
        )


    predictions = (
        pd.concat(pred_rows, ignore_index=True)
        .sort_values(["load_area", "timestamp_local"])
        .reset_index(drop=True)
    )

    corrected_predictions = apply_projection_correction(predictions)
    if "mw_pred_corrected" in corrected_predictions.columns and PEAK_RANGES:
        def _clip_group(group: pd.DataFrame) -> pd.DataFrame:
            bounds = PEAK_RANGES.get(group.name)
            if not bounds:
                return group
            lower, upper = bounds
            group["mw_pred_corrected"] = group["mw_pred_corrected"].clip(lower=lower, upper=upper)
            return group

        corrected_predictions = (
            corrected_predictions.groupby("load_area", group_keys=False).apply(_clip_group)
        )
    vprint(f"Predicting for {day.date()}:")

    LAG_HOURS = [48, 72, 96]  # keep in sync with training config

    for area, area_preds in corrected_predictions.groupby("load_area", sort=False):
        value_col = "mw_pred_corrected" if "mw_pred_corrected" in area_preds.columns else "mw_pred"
        area_preds = area_preds.sort_values("timestamp_local").reset_index(drop=True)

        if area not in DATASETS:
            base_df = area_preds[["timestamp_local", value_col]].rename(columns={value_col: "mw"})
            base_df["load_area"] = area
            for lag in LAG_HOURS:
                base_df[f"lag_{lag}"] = np.nan
            DATASETS[area] = (base_df, None)
            continue

        full_area_df, feature_cols = DATASETS[area]
        full_area_df = full_area_df.sort_values("timestamp_local").reset_index(drop=True)
        for lag in LAG_HOURS:
            if f"lag_{lag}" not in full_area_df.columns:
                full_area_df[f"lag_{lag}"] = np.nan

        def ensure_row(ts: pd.Timestamp) -> pd.Series:
            mask = full_area_df["timestamp_local"] == ts
            if mask.any():
                return mask
            new_row = {col: np.nan for col in full_area_df.columns}
            new_row["timestamp_local"] = ts
            new_row["load_area"] = area
            full_area_df.loc[len(full_area_df)] = new_row
            return full_area_df["timestamp_local"] == ts

        for _, row in area_preds.iterrows():
            ts = row["timestamp_local"]
            pred_val = row[value_col]

            mask_current = ensure_row(ts)
            full_area_df.loc[mask_current, "mw"] = pred_val

            for lag in LAG_HOURS:
                future_ts = ts + pd.Timedelta(hours=lag)
                mask_future = ensure_row(future_ts)
                full_area_df.loc[mask_future, f"lag_{lag}"] = pred_val

        full_area_df = full_area_df.sort_values("timestamp_local").reset_index(drop=True)
        DATASETS[area] = (full_area_df, feature_cols)


# Final output for the target day
CURRENT_DAY_LOAD_PREDICTIONS = corrected_predictions

peak_lookup = {
    "AECO": ["2025-11-23", "2025-11-21"],
    "AEPAPT": ["2025-11-25", "2025-11-21"],
    "AEPIMP": ["2025-11-25", "2025-11-21"],
    "AEPKPT": ["2025-11-25", "2025-11-21"],
    "AEPOPT": ["2025-11-25", "2025-11-20"],
    "AP": ["2025-11-21", "2025-11-28"],
    "BC": ["2025-11-21", "2025-11-28"],
    "CE": ["2025-11-25", "2025-11-22"],
    "DAY": ["2025-11-25", "2025-11-21"],
    "DEOK": ["2025-11-25", "2025-11-21"],
    "DOM": ["2025-11-25", "2025-11-21"],
    "DPLCO": ["2025-11-21", "2025-11-28"],
    "DUQ": ["2025-11-21", "2025-11-25"],
    "EASTON": ["2025-11-21", "2025-11-28"],
    "EKPC": ["2025-11-27", "2025-11-20"],
    "JC": ["2025-11-25", "2025-11-21"],
    "ME": ["2025-11-25", "2025-11-21"],
    "OE": ["2025-11-21", "2025-11-25"],
    "OVEC": ["2025-11-21", "2025-11-25"],
    "PAPWR": ["2025-11-25", "2025-11-20"],
    "PE": ["2025-11-23", "2025-11-21"],
    "PEPCO": ["2025-11-20", "2025-11-21"],
    "PLCO": ["2025-11-25", "2025-11-21"],
    "PN": ["2025-11-25", "2025-11-21"],
    "PS": ["2025-11-25", "2025-11-21"],
    "RECO": ["2025-11-25", "2025-11-19"],
    "RTO": ["2025-11-20", "2025-11-21"],
    "SMECO": ["2025-11-22", "2025-11-21"],
    "UGI": ["2025-11-25", "2025-11-21"],
    "VMEU": ["2025-11-20", "2025-11-21"],
}

peak_flags = [
    "1" if TARGET_DAY in peak_lookup.get(area, []) else "0"
    for area in AREAS
]

peak_flags = [
    "1" if TARGET_DAY in peak_lookup.get(area, []) else "0"
    for area in AREAS
]

# hour (0–23) of the max corrected prediction for each area
peak_hours = []
for area in AREAS:
    area_preds = CURRENT_DAY_LOAD_PREDICTIONS[CURRENT_DAY_LOAD_PREDICTIONS["load_area"] == area]
    if area_preds.empty:
        peak_hours.append("0")
    else:
        value_col = "mw_pred_corrected" if "mw_pred_corrected" in area_preds.columns else "mw_pred"
        peak_row = area_preds.loc[area_preds[value_col].idxmax()]
        peak_hours.append(str(peak_row["timestamp_local"].hour))

print(f"{TARGET_DAY}, ", end="")
print(", ".join(f"{v:.2f}" for v in CURRENT_DAY_LOAD_PREDICTIONS["mw_pred_corrected"]), end="")
print(", " + ", ".join(peak_hours), end="")
print(", " + ", ".join(peak_flags))
