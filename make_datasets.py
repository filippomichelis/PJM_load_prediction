import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

out_path = Path("data/full_datasets.joblib")

LOAD_DIR = Path('..')
WEATHER_CSV = 'data/weather_points_oct_nov_dec.csv'
OUTPUT_CSV = Path('..') / 'load_predictions_custom.csv'
TRAIN_END = pd.Timestamp('2025-10-20 23:00:00', tz='America/New_York')
TEST_START = pd.Timestamp('2025-10-21 00:00:00', tz='America/New_York')
TEST_END = pd.Timestamp('2025-10-30 00:00:00', tz='America/New_York')
KEEP_YEARS = {2019, 2022, 2023, 2024, 2025}
KEEP_MONTHS = {10, 11, 12}
LAG_HOURS = [48, 72, 96]
TZ_NAME = "America/New_York"

VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# Thanksgiving helpers
def _thanksgiving_dates(years, tz_name: str):
    dates = set()
    for year in map(int, years):
        date = pd.Timestamp(year=year, month=11, day=1, tz=tz_name)
        while date.weekday() != 3:
            date += pd.Timedelta(days=1)
        date += pd.Timedelta(days=21)
        dates.add(date.normalize())
    return dates


def add_thanksgiving_flag(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if df.empty:
        df["is_thanksgiving"] = pd.Series(dtype="int8")
        return df
    tg_dates = _thanksgiving_dates(df["timestamp_local"].dt.year.unique(), tz_name)
    df["is_thanksgiving"] = (
        df["timestamp_local"].dt.normalize().isin(tg_dates).astype("int8")
    )
    return df

# Selected weather columns to use
WEATHER_COLS = [
    "temp_c",
    "dewpoint_c",
    "humidity_pct",
    "precip_mm",
    "wind_speed_ms",
    "cooling_deg_f",
    "heating_deg_f",
]

def build_region_weather_datasets(weather_csv, out_dir):
    weather_csv = Path(weather_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(
        weather_csv,
        usecols=["load_area", "point_id", "timestamp_local", *WEATHER_COLS],
    )
    weather["timestamp_local"] = (
        pd.to_datetime(weather["timestamp_local"], utc=True, errors="coerce")
        .dt.tz_convert(TZ_NAME)
    )
    weather = weather.dropna(subset=["timestamp_local", "point_id", "load_area"])

    point_lookup = (
        weather[["load_area", "point_id"]]
        .drop_duplicates()
        .sort_values(["load_area", "point_id"])
        .assign(point_num=lambda df: df.groupby("load_area").cumcount() + 1)
    )
    weather = weather.merge(point_lookup, on=["load_area", "point_id"], how="left")
    weather = (
        weather.sort_values(["load_area", "timestamp_local", "point_num"])
        .drop_duplicates(["load_area", "timestamp_local", "point_num"])
    )

    stacked = (
        weather.set_index(["load_area", "timestamp_local", "point_num"])[WEATHER_COLS]
        .unstack("point_num")
        .sort_index(axis=1, level=1)
    )
    feature_lookup = {name: idx + 1 for idx, name in enumerate(WEATHER_COLS)}
    stacked.columns = [
        f"weather_feature{feature_lookup[col_name]}_point_{int(point_num)}"
        for (col_name, point_num) in stacked.columns
    ]
    stacked = stacked.reset_index()

    out_paths = {}
    for load_area, area_df in stacked.groupby("load_area", sort=False):
        area_df = area_df.drop(columns=["load_area"]).reset_index(drop=True)
        keep_cols = [
            col
            for col in area_df.columns
            if col == "timestamp_local" or area_df[col].notna().any()
        ]
        cleaned = area_df.loc[:, keep_cols]
        area_path = out_dir / f"weather_{load_area}.csv"
        cleaned.to_csv(area_path, index=False)
        out_paths[load_area] = area_path

    vprint(f"Saved {len(out_paths)} per-load-area weather files to {out_dir}")
    return stacked, out_paths

def build_full_datasets_by_area(load_df, region_weather_paths):
    """Return {load_area: (full_df, feature_cols)} ready for training."""
    base_feature_cols = (
        [f"lag_{lag}" for lag in LAG_HOURS]
        + ["hour", "dow", "dayofyear", "month", "is_weekend", "is_thanksgiving"]
    )

    area_frames = {}
    for load_area, weather_path in region_weather_paths.items():
        area_load = load_df[load_df["load_area"] == load_area].copy()
        weather_df = pd.read_csv(weather_path)
        weather_df["timestamp_local"] = (
            pd.to_datetime(weather_df["timestamp_local"], utc=True, errors="coerce")
            .dt.tz_convert(TZ_NAME)
        )
        weather_df["load_area"] = load_area

        if area_load.empty:
            combined_index = pd.DatetimeIndex(weather_df["timestamp_local"]).sort_values()
            area_load = pd.DataFrame({"timestamp_local": combined_index})
            area_load["load_area"] = load_area
            for lag in LAG_HOURS:
                area_load[f"lag_{lag}"] = np.nan
            area_load["mw"] = np.nan
        else:
            load_index = pd.DatetimeIndex(area_load["timestamp_local"])
            weather_index = pd.DatetimeIndex(weather_df["timestamp_local"])
            combined_index = load_index.union(weather_index).sort_values()

            area_load = (
                area_load.set_index("timestamp_local")
                .reindex(combined_index)
                .reset_index()
                .rename(columns={"index": "timestamp_local"})
            )
            area_load["load_area"] = load_area

        area_load = area_load.sort_values("timestamp_local").reset_index(drop=True)
        for lag in LAG_HOURS:
            area_load[f"lag_{lag}"] = area_load["mw"].shift(lag)

        # recompute basic temporal features for any newly added timestamps
        area_load["hour"] = area_load["timestamp_local"].dt.hour
        area_load["dow"] = area_load["timestamp_local"].dt.dayofweek
        area_load["dayofyear"] = area_load["timestamp_local"].dt.dayofyear
        area_load["month"] = area_load["timestamp_local"].dt.month
        area_load["is_weekend"] = area_load["dow"].isin([5, 6]).astype(int)
        area_load = add_thanksgiving_flag(area_load, TZ_NAME)

        # merge weather features (ensures dataset runs through latest weather timestamp)
        full_df = area_load.merge(
            weather_df,
            on=["load_area", "timestamp_local"],
            how="left",
            validate="1:1",
        )

        weather_feature_cols = [
            c
            for c in weather_df.columns
            if c not in {"load_area", "timestamp_local"}
        ]
        feature_cols = base_feature_cols + weather_feature_cols
        area_frames[load_area] = (full_df, feature_cols)

    return area_frames

def load_load_history(load_dir: Path, tz_name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(load_dir.glob("hrl_load_metered_*.csv")):
        year = int(path.stem[-4:])
        if year not in KEEP_YEARS:
            continue
        df = pd.read_csv(
            path,
            usecols=["datetime_beginning_ept", "load_area", "mw"],
            parse_dates=["datetime_beginning_ept"],
        )
        df = df.rename(columns={"datetime_beginning_ept": "timestamp_local"})
        df["timestamp_local"] = (
            df["timestamp_local"]
            .dt.tz_localize(tz_name, nonexistent="shift_forward", ambiguous="NaT")
        )
        df = df.dropna(subset=["timestamp_local"])
        df = df[df["timestamp_local"].dt.month.isin(KEEP_MONTHS)]
        frames.append(df)

    if not frames:
        raise RuntimeError("No load files found for requested years/months.")

    load_df = pd.concat(frames, ignore_index=True)
    load_df = load_df.sort_values(["load_area", "timestamp_local"]).reset_index(drop=True)
    for lag in LAG_HOURS:
        load_df[f"lag_{lag}"] = load_df.groupby("load_area")["mw"].shift(lag)
    load_df["hour"] = load_df["timestamp_local"].dt.hour
    load_df["dow"] = load_df["timestamp_local"].dt.dayofweek
    load_df["dayofyear"] = load_df["timestamp_local"].dt.dayofyear
    load_df["month"] = load_df["timestamp_local"].dt.month
    load_df["is_weekend"] = load_df["dow"].isin([5, 6]).astype(int)
    load_df = add_thanksgiving_flag(load_df, TZ_NAME)
    return load_df

# Load full weather dataset
weather_points = pd.read_csv(WEATHER_CSV)

# Build region weather datasets
REGION_WEATHER_OUT = Path("weather_by_load_area")
region_weather_full, region_weather_paths = build_region_weather_datasets(
    WEATHER_CSV,
    REGION_WEATHER_OUT,
)

# Load load history dataset
load_dir = Path("data")
load_df = load_load_history(load_dir, TZ_NAME)

# Do not drop rows missing lags; future timestamps will carry NaNs for unavailable lag inputs.
# lag columns already created inside load_load_history.

# Build every area’s full_df + feature list
full_datasets = build_full_datasets_by_area(load_df, region_weather_paths)
out_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(full_datasets, out_path)
print(f"Saved full_datasets to {out_path.resolve()}")
