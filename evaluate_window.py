#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
KEEP_YEARS = {2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025}
KEEP_MONTHS = {10, 11, 12}
LAG_HOURS = [24, 48, 72, 96]
DEFAULT_TZ = "America/New_York"
WEATHER_COLS = [
    "temp_c",
    "dewpoint_c",
    "humidity_pct",
    "precip_mm",
    "snow_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "sunshine_minutes",
    "cooling_deg_f",
    "heating_deg_f",
]


class RegionModelPayload:
    def __init__(self, load_area: str, pipeline, kept_features: List[str]):
        self.load_area = load_area
        self.pipeline = pipeline
        self.kept_features = kept_features


def load_residual_stacker(model_dir: Path) -> dict | None:
    path = Path(model_dir) / "residual_stacker.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Data prep helpers (copied from train/predict scripts)
# ---------------------------------------------------------------------------
def load_load_history(load_dir: Path, tz_name: str) -> pd.DataFrame:
    load_dir = Path(load_dir)
    frames = []
    for path in sorted(load_dir.glob("hrl_load_metered_*.csv")):
        try:
            year = int(path.stem[-4:])
        except ValueError:
            continue
        if year not in KEEP_YEARS:
            continue
        df = pd.read_csv(
            path,
            usecols=["datetime_beginning_ept", "load_area", "mw"],
        )
        df["datetime_beginning_ept"] = pd.to_datetime(
            df["datetime_beginning_ept"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        df = df.rename(columns={"datetime_beginning_ept": "timestamp_local"})
        df["timestamp_local"] = df["timestamp_local"].dt.tz_localize(
            tz_name, nonexistent="shift_forward", ambiguous="NaT"
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
    return load_df


def load_weather(weather_csv: Path, tz_name: str) -> pd.DataFrame:
    weather = pd.read_csv(
        weather_csv,
        usecols=["load_area", "timestamp_local", *WEATHER_COLS],
    )
    weather["timestamp_local"] = pd.to_datetime(
        weather["timestamp_local"], utc=True, errors="coerce"
    ).dt.tz_convert(tz_name)
    weather = weather.dropna(subset=["timestamp_local"])
    agg = (
        weather.groupby(["load_area", "timestamp_local"])[WEATHER_COLS]
        .mean()
        .reset_index()
    )
    return agg


def build_full_datasets_by_area(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    base_feature_cols = (
        [f"lag_{lag}" for lag in LAG_HOURS]
        + ["hour", "dow", "dayofyear", "month", "is_weekend"]
    )

    datasets: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
    weather_feature_cols = [
        c for c in weather_df.columns if c not in {"load_area", "timestamp_local"}
    ]
    for load_area, area_load in load_df.groupby("load_area"):
        area_weather = weather_df[weather_df["load_area"] == load_area]
        full_df = area_load.merge(
            area_weather,
            on=["load_area", "timestamp_local"],
            how="left",
            validate="1:1",
        )
        feature_cols = base_feature_cols + weather_feature_cols
        datasets[load_area] = (full_df, feature_cols)

    if not datasets:
        raise RuntimeError("Failed to build any load-area datasets. Check inputs.")
    return datasets


def load_or_build_full_datasets(
    dataset_path: Path | None,
    cache_path: Path | None,
    load_dir: Path,
    weather_csv: Path,
    tz_name: str,
) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    dataset_path = Path(dataset_path) if dataset_path else None
    if dataset_path and dataset_path.exists():
        print(f"Loading cached full_datasets from {dataset_path}")
        return joblib.load(dataset_path)

    print("full_datasets cache not found; building from raw load + weather inputs.")
    load_df = load_load_history(Path(load_dir), tz_name)
    weather_df = load_weather(Path(weather_csv), tz_name)
    datasets = build_full_datasets_by_area(load_df, weather_df)
    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(datasets, cache_path)
        print(f"Wrote full_datasets cache to {cache_path}")
    return datasets


def _thanksgiving_dates(years: np.ndarray, tz: str) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for year in map(int, years):
        date = pd.Timestamp(year=year, month=11, day=1, tz=tz)
        while date.weekday() != 3:
            date += pd.Timedelta(days=1)
        dates.add((date + pd.Timedelta(days=21)).normalize())
    return dates


def _add_thanksgiving_flag(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["is_thanksgiving"] = pd.Series(dtype="int8")
        return df
    tg_dates = _thanksgiving_dates(df["timestamp_local"].dt.year.unique(), tz=tz)
    df["is_thanksgiving"] = df["timestamp_local"].dt.normalize().isin(tg_dates).astype("int8")
    return df


def evaluate_region_models(
    region_models: List[RegionModelPayload],
    datasets: Dict[str, Tuple[pd.DataFrame, List[str]]],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    tz_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    records = []
    mse_by_region: Dict[str, float] = {}
    for rm in region_models:
        if rm.load_area not in datasets:
            continue
        area_df, _ = datasets[rm.load_area]
        window = area_df[
            (area_df["timestamp_local"] >= test_start)
            & (area_df["timestamp_local"] < test_end)
        ].copy()
        if window.empty:
            continue
        window = _add_thanksgiving_flag(window, tz=tz_name)

        missing = [col for col in rm.kept_features if col not in window.columns]
        for col in missing:
            window[col] = np.nan

        X_test = window[rm.kept_features]
        y_true = window["mw"]
        y_pred = rm.pipeline.predict(X_test)

        mse = mean_squared_error(y_true, y_pred)
        mse_by_region[rm.load_area] = mse
        records.extend(
            {
                "load_area": rm.load_area,
                "timestamp_local": ts,
                "mw_actual": actual,
                "mw_pred": pred,
                "error": pred - actual,
            }
            for ts, actual, pred in zip(window["timestamp_local"], y_true, y_pred)
        )

    if records:
        predictions_df = pd.DataFrame(records).sort_values(["load_area", "timestamp_local"])
    else:
        predictions_df = pd.DataFrame(columns=["load_area", "timestamp_local", "mw_actual", "mw_pred", "error"])
    overall_mse = (
        mean_squared_error(predictions_df["mw_actual"], predictions_df["mw_pred"])
        if not predictions_df.empty
        else float("nan")
    )
    return predictions_df, mse_by_region, overall_mse


def build_stacker_inputs(preds_df: pd.DataFrame, stack_feature_cols: List[str]) -> pd.DataFrame:
    if preds_df.empty:
        return pd.DataFrame()

    preds_df = preds_df.sort_values("timestamp_local")
    residual_matrix = preds_df.pivot_table(
        index="timestamp_local", columns="load_area", values="error"
    ).sort_index()
    if residual_matrix.empty:
        return pd.DataFrame()

    n_areas = residual_matrix.shape[1]
    residual_mean_all = residual_matrix.mean(axis=1).fillna(0.0)
    residual_std_all = residual_matrix.std(axis=1).fillna(0.0)
    filled_matrix = residual_matrix.fillna(0.0)

    pc_cols = [col for col in stack_feature_cols if col.startswith("res_pc")]
    pc_df = pd.DataFrame(0.0, index=residual_matrix.index, columns=pc_cols)
    n_components = min(len(pc_cols), filled_matrix.shape[0], filled_matrix.shape[1])
    if n_components > 0:
        pcs = PCA(n_components=n_components, random_state=0).fit_transform(filled_matrix.values)
        for idx in range(n_components):
            pc_df.loc[:, pc_cols[idx]] = pcs[:, idx]

    lag1 = residual_matrix.shift(1).fillna(0.0)
    lag24 = residual_matrix.shift(24).fillna(0.0)

    preds_lookup = preds_df.set_index(["timestamp_local", "load_area"])
    rows = []
    for ts in residual_matrix.index:
        ts_errors = residual_matrix.loc[ts]
        for area in residual_matrix.columns:
            key = (ts, area)
            if key not in preds_lookup.index:
                continue
            record = preds_lookup.loc[key]
            residual = ts_errors.get(area, 0.0)
            residual = 0.0 if pd.isna(residual) else residual
            if n_areas > 1:
                others_mean = (residual_mean_all.loc[ts] * n_areas - residual) / (n_areas - 1)
            else:
                others_mean = 0.0
            row = {
                "load_area": area,
                "timestamp_local": ts,
                "base_pred": record["mw_pred"],
                "residual_same_area": residual,
                "residual_mean_all": residual_mean_all.loc[ts],
                "residual_mean_others": others_mean,
                "residual_std_all": residual_std_all.loc[ts],
                "lag1_residual": lag1.loc[ts, area],
                "lag24_residual": lag24.loc[ts, area],
                "hour": ts.hour,
                "dow": ts.dayofweek,
            }
            for col in pc_cols:
                row[col] = pc_df.loc[ts, col]
            rows.append(row)
    stack_df = pd.DataFrame(rows)
    if stack_df.empty:
        return stack_df

    for col in stack_feature_cols:
        if col not in stack_df.columns:
            stack_df[col] = 0.0
    return stack_df[["load_area", "timestamp_local", *stack_feature_cols]]


def apply_residual_stacker(preds_df: pd.DataFrame, stacker_meta: dict | None) -> pd.DataFrame:
    if stacker_meta is None or preds_df.empty:
        return preds_df
    stack_features = build_stacker_inputs(preds_df, stacker_meta["feature_columns"])
    if stack_features.empty:
        return preds_df
    pipeline = stacker_meta["pipeline"]
    X = stack_features[stacker_meta["feature_columns"] + ["load_area"]]
    corrections = pipeline.predict(X)
    stack_features = stack_features.assign(residual_correction=corrections)
    preds_df = preds_df.merge(
        stack_features[["load_area", "timestamp_local", "residual_correction"]],
        on=["load_area", "timestamp_local"],
        how="left",
    )
    preds_df["residual_correction"] = preds_df["residual_correction"].fillna(0.0)
    preds_df["mw_pred"] = preds_df["mw_pred"] + preds_df["residual_correction"]
    preds_df["error"] = preds_df["mw_pred"] - preds_df["mw_actual"]
    return preds_df


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved region models over a date range and report MSE.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"), help="Directory containing region_models.joblib.")
    parser.add_argument(
        "--full-datasets",
        type=Path,
        default=Path("artifacts/full_datasets.joblib"),
        help="Cached full_datasets joblib (rebuilt if missing).",
    )
    parser.add_argument(
        "--no-cache-full-datasets",
        action="store_true",
        help="Do not write the rebuilt full_datasets cache.",
    )
    parser.add_argument("--load-dir", type=Path, default=Path("data"), help="Directory with hrl_load_metered_*.csv files.")
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=Path("data/weather_points_oct_nov_dec.csv"),
        help="Aggregated weather CSV.",
    )
    parser.add_argument("--tz", type=str, default=DEFAULT_TZ, help="IANA timezone.")
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Evaluation window start (inclusive, e.g. 2025-10-21 00:00:00).",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Evaluation window end (exclusive).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Optional path to write the per-timestamp prediction records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_blob = joblib.load(Path(args.model_dir) / "region_models.joblib")
    region_models = [
        RegionModelPayload(
            load_area=area,
            pipeline=payload["pipeline"],
            kept_features=payload["kept_features"],
        )
        for area, payload in model_blob.items()
    ]
    if not region_models:
        raise RuntimeError("No region models found in artifact payload.")

    stacker_meta = load_residual_stacker(args.model_dir)
    if stacker_meta is None:
        print("Warning: residual stacker artifact not found; evaluation will use base model outputs.")

    cache_target = None if args.no_cache_full_datasets else args.full_datasets
    datasets = load_or_build_full_datasets(
        dataset_path=args.full_datasets,
        cache_path=cache_target,
        load_dir=args.load_dir,
        weather_csv=args.weather_csv,
        tz_name=args.tz,
    )

    start_ts = pd.Timestamp(args.start, tz=args.tz)
    end_ts = pd.Timestamp(args.end, tz=args.tz)

    preds_df, mse_by_region, overall_mse = evaluate_region_models(
        region_models,
        datasets,
        test_start=start_ts,
        test_end=end_ts,
        tz_name=args.tz,
    )
    preds_df = apply_residual_stacker(preds_df, stacker_meta)

    if preds_df.empty:
        print("No predictions generated for the specified window.")
    else:
        overall_mse = mean_squared_error(preds_df["mw_actual"], preds_df["mw_pred"])
        print(f"Overall MSE: {overall_mse:,.2f}")
        summary = (
            preds_df.groupby("load_area")
            .apply(lambda df: mean_squared_error(df["mw_actual"], df["mw_pred"]))
            .rename("mse")
            .sort_values()
        )
        print("\nPer-load-area MSE:")
        print(summary)

    if args.out_csv:
        preds_df.to_csv(args.out_csv, index=False)
        print(f"Wrote detailed predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
