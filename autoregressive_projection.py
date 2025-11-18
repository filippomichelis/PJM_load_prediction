#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from predict_day import thanksgiving_flag
from projection_stacker import apply_projection_correction, train_projection_for_window

MODEL_DIR = Path("artifacts")
FULL_DS_PATH = Path("data/full_datasets.joblib")
LOAD_HISTORY_CSV = Path("data/hrl_load_metered_2025.csv")
TZ_NAME = "America/New_York"

AREAS = [
    "AECO",
    "AEPAPT",
    "AEPIMP",
    "AEPKPT",
    "AEPOPT",
    "AP",
    "BC",
    "CE",
    "DAY",
    "DEOK",
    "DOM",
    "DPLCO",
    "DUQ",
    "EASTON",
    "EKPC",
    "JC",
    "ME",
    "OE",
    "OVEC",
    "PAPWR",
    "PE",
    "PEPCO",
    "PLCO",
    "PN",
    "PS",
    "RECO",
    "SMECO",
    "UGI",
    "VMEU",
]


def _load_region_payloads(model_dir: Path) -> Dict[str, dict]:
    path = Path(model_dir) / "region_models.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing region model artifact: {path}")
    payload = joblib.load(path)
    return payload


def _load_full_datasets(cache_path: Path) -> Dict[str, tuple[pd.DataFrame, List[str]]]:
    cache = Path(cache_path)
    if not cache.exists():
        raise FileNotFoundError(f"Full datasets cache missing: {cache}")
    return joblib.load(cache)


def _load_history(load_csv: Path, tz_name: str) -> Dict[str, pd.DataFrame]:
    history = pd.read_csv(
        load_csv,
        usecols=["datetime_beginning_ept", "load_area", "mw"],
        parse_dates=["datetime_beginning_ept"],
    ).rename(columns={"datetime_beginning_ept": "timestamp_local"})
    history["timestamp_local"] = history["timestamp_local"].dt.tz_localize(
        tz_name, nonexistent="shift_forward", ambiguous="NaT"
    )
    history = history.dropna(subset=["timestamp_local"])
    return {area: df.reset_index(drop=True) for area, df in history.groupby("load_area", sort=False)}


def average_last_three_days(area_df: pd.DataFrame, day_start: pd.Timestamp) -> pd.DataFrame:
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

    target_hours = pd.date_range(day_start, periods=24, freq="H", tz=day_start.tz)
    values = [hourly.get(h, fallback) for h in range(24)]
    area = area_df["load_area"].iloc[0] if not area_df.empty else "UNKNOWN"
    return pd.DataFrame(
        {
            "load_area": area,
            "timestamp_local": target_hours,
            "mw_pred": values,
            "prediction_source": "average",
        }
    )


def _infer_lag_hours(feature_cols: List[str]) -> List[int]:
    lag_hours: List[int] = []
    for col in feature_cols:
        if not col.startswith("lag_"):
            continue
        try:
            lag_hours.append(int(col.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(lag_hours)


def autoregressive_forecast_area(
    area: str,
    payload: dict,
    area_df: pd.DataFrame,
    feature_cols: List[str],
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    tz_name: str,
) -> pd.DataFrame:
    if area_df.empty:
        return pd.DataFrame()

    area_df = area_df.sort_values("timestamp_local").reset_index(drop=True)
    history_values = area_df.set_index("timestamp_local")["mw"].dropna()
    if history_values.empty:
        return pd.DataFrame()

    horizon_end = day_end - pd.Timedelta(hours=1)
    last_history_ts = history_values.index.max()

    if last_history_ts >= horizon_end:
        mask = (area_df["timestamp_local"] >= day_start) & (area_df["timestamp_local"] <= horizon_end)
        actual = area_df.loc[mask, ["timestamp_local", "mw"]].dropna()
        if actual.empty:
            return pd.DataFrame()
        pred = actual.assign(load_area=area, mw_pred=actual["mw"], prediction_source="actual")
        return pred.drop(columns="mw")

    area_future = area_df[area_df["timestamp_local"] > last_history_ts]
    if area_future.empty:
        return pd.DataFrame()

    future_inputs = area_future.set_index("timestamp_local")
    lag_hours = _infer_lag_hours(feature_cols)
    base_cols = [col for col in feature_cols if col not in {f"lag_{lag}" for lag in lag_hours}]

    preds = []
    ts = last_history_ts + pd.Timedelta(hours=1)
    while ts <= horizon_end:
        if ts not in future_inputs.index:
            break
        row = future_inputs.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        features = {col: row.get(col, np.nan) for col in base_cols}
        if "hour" in feature_cols:
            features["hour"] = ts.hour
        if "dow" in feature_cols:
            features["dow"] = ts.dayofweek
        if "dayofyear" in feature_cols:
            features["dayofyear"] = ts.dayofyear
        if "month" in feature_cols:
            features["month"] = ts.month
        if "is_weekend" in feature_cols:
            features["is_weekend"] = int(ts.dayofweek in (5, 6))
        if "is_thanksgiving" in feature_cols:
            features["is_thanksgiving"] = thanksgiving_flag(ts, tz_name)

        missing_lag = False
        for lag in lag_hours:
            lag_col = f"lag_{lag}"
            lag_ts = ts - pd.Timedelta(hours=lag)
            val = history_values.loc[lag_ts] if lag_ts in history_values.index else np.nan
            if pd.isna(val):
                missing_lag = True
                break
            features[lag_col] = val
        if missing_lag:
            break

        feature_df = pd.DataFrame([features])[feature_cols]
        pred = payload["pipeline"].predict(feature_df)[0]
        if ts >= day_start:
            preds.append({"load_area": area, "timestamp_local": ts, "mw_pred": pred, "prediction_source": "model"})
        history_values.loc[ts] = pred
        history_values = history_values.sort_index()
        ts += pd.Timedelta(hours=1)

    return pd.DataFrame(preds)


def generate_autoregressive_predictions(
    target_day: str,
    stack_window_start: str = "2025-10-16 00:00:00",
    stack_window_end: str = "2025-10-21 00:00:00",
) -> pd.DataFrame:
    region_payload = _load_region_payloads(MODEL_DIR)
    datasets = _load_full_datasets(FULL_DS_PATH)
    fallback_history = _load_history(LOAD_HISTORY_CSV, TZ_NAME)

    day_start = pd.Timestamp(target_day, tz=TZ_NAME)
    day_end = day_start + pd.Timedelta(days=1)

    train_projection_for_window(
        window_start=stack_window_start,
        window_end=stack_window_end,
        model_dir=MODEL_DIR,
        full_datasets_path=FULL_DS_PATH,
    )

    pred_rows = []
    for area in AREAS:
        if area in region_payload and area in datasets:
            area_df, feature_cols = datasets[area]
            preds = autoregressive_forecast_area(
                area=area,
                payload=region_payload[area],
                area_df=area_df,
                feature_cols=feature_cols,
                day_start=day_start,
                day_end=day_end,
                tz_name=TZ_NAME,
            )
            if preds.empty:
                print(f"Warning: autoregressive forecast failed for {area}; using fallback averages.")
            else:
                pred_rows.append(preds)
                continue

        if area in fallback_history:
            avg_df = average_last_three_days(fallback_history[area], day_start)
            if avg_df.empty:
                print(f"Warning: insufficient fallback history for {area}.")
                continue
            pred_rows.append(avg_df)
        else:
            print(f"Warning: no dataset or history available for {area}.")

    if not pred_rows:
        raise RuntimeError("Failed to produce any predictions.")

    predictions = (
        pd.concat(pred_rows, ignore_index=True)
        .sort_values(["load_area", "timestamp_local"])
        .reset_index(drop=True)
    )

    corrected = apply_projection_correction(
        predictions,
        apply_mask=predictions["prediction_source"].eq("model"),
    )
    return corrected


if __name__ == "__main__":
    target = "2025-11-13"
    preds = generate_autoregressive_predictions(target)
    print(f"Generated {len(preds)} rows for {target}.")
    print(preds.head())
