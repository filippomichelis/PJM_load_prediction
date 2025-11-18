#!/usr/bin/env python3
"""
Train per-load-area models using 4 prior days of load plus same-day weather,
then predict hour-by-hour load for Nov 1-9, 2025.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

KEEP_YEARS = {2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025}
KEEP_MONTHS = {10, 11, 12}
LAG_HOURS = [24, 48, 72, 96]
TZ_NAME = "America/New_York"
PRED_START = pd.Timestamp("2025-11-01 00:00:00", tz=TZ_NAME)
PRED_END = pd.Timestamp("2025-11-10 00:00:00", tz=TZ_NAME)
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


def parse_args():
    p = argparse.ArgumentParser(description="Train/predict PJM load using lag features + weather.")
    p.add_argument(
        "--load-dir",
        type=str,
        default=".",
        help="Directory containing hrl_load_metered_YYYY.csv files",
    )
    p.add_argument(
        "--weather-csv",
        type=str,
        default="weather_points_oct_nov_dec.csv",
        help="Aggregated weather dataset (months Oct-Dec)",
    )
    p.add_argument(
        "--points-csv",
        type=str,
        default="pjm_load_area_cities(1).csv",
        help="Points metadata (used to validate load areas)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="load_predictions_nov1_9_2025.csv",
        help="Output CSV for predictions",
    )
    return p.parse_args()


def load_load_history(load_dir: Path) -> pd.DataFrame:
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
            .dt.tz_localize(TZ_NAME, nonexistent="shift_forward", ambiguous="NaT")
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


def load_weather(weather_csv: Path) -> pd.DataFrame:
    weather = pd.read_csv(
        weather_csv,
        usecols=["load_area", "timestamp_local", *WEATHER_COLS],
    )
    weather["timestamp_local"] = pd.to_datetime(
        weather["timestamp_local"], utc=True, errors="coerce"
    ).dt.tz_convert(TZ_NAME)
    weather = weather.dropna(subset=["timestamp_local"])
    agg = (
        weather.groupby(["load_area", "timestamp_local"])[WEATHER_COLS]
        .mean()
        .reset_index()
    )
    return agg


def train_area_model(
    area_df: pd.DataFrame,
    feature_cols: list[str],
    pred_start: pd.Timestamp,
) -> dict:
    area_name = area_df["load_area"].iloc[0]
    area_df = area_df.dropna(subset=[f"lag_{lag}" for lag in LAG_HOURS])
    train_df = area_df[(area_df["timestamp_local"] < pred_start) & area_df["mw"].notna()]
    if len(train_df) < 200:
        return {}

    train_X = train_df[feature_cols]
    fill_values = train_X.median()
    train_X = train_X.fillna(fill_values)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=500, random_state=0
    )
    model.fit(train_X, train_df["mw"])

    history_values = (
        train_df[["timestamp_local", "mw"]]
        .set_index("timestamp_local")["mw"]
        .sort_index()
    )
    history_flags = pd.Series(False, index=history_values.index)

    return {
        "model": model,
        "fill_values": fill_values,
        "history_values": history_values,
        "history_flags": history_flags,
        "load_area": area_name,
    }


def predict_area(
    model_state: dict,
    feature_cols: list[str],
    input_df: pd.DataFrame,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    verbose: bool = True,
) -> pd.DataFrame:
    if not model_state:
        return pd.DataFrame()

    model = model_state["model"]
    fill_values = model_state["fill_values"]
    history_values = model_state["history_values"].copy()
    history_flags = model_state["history_flags"].copy()
    area_name = model_state["load_area"]
    warning_dates = set()

    area_inputs = input_df[input_df["load_area"] == area_name]
    if area_inputs.empty:
        if verbose:
            print(f"[warning] No weather rows supplied for {area_name} in prediction window.")
        return pd.DataFrame()
    area_inputs = area_inputs.set_index("timestamp_local").sort_index()
    future_index = pd.date_range(
        start=pred_start,
        end=pred_end - pd.Timedelta(hours=1),
        freq="H",
        tz=TZ_NAME,
    )

    preds = []
    for ts in future_index:
        feature = {}
        missing_lag = False
        for lag in LAG_HOURS:
            lag_ts = ts - pd.Timedelta(hours=lag)
            if lag_ts in history_values.index:
                val = history_values.loc[lag_ts]
                flag = history_flags.loc[lag_ts]
            else:
                pos = history_values.index.searchsorted(lag_ts)
                if pos == 0:
                    val = np.nan
                    flag = True
                else:
                    val = history_values.iloc[pos - 1]
                    flag = history_flags.iloc[pos - 1]
            if pd.isna(val):
                missing_lag = True
                break

            feature[f"lag_{lag}"] = val
            if flag and verbose and ts.date() not in warning_dates:
                print(
                    f"[warning] Using predicted lag inputs for {area_name} on {ts.date()} (lag {lag}h)"
                )
                warning_dates.add(ts.date())

        if missing_lag:
            break

        if ts not in area_inputs.index:
            if verbose:
                print(f"[warning] Missing weather inputs for {area_name} at {ts}")
            break
        weather_vals = area_inputs.loc[ts, WEATHER_COLS]

        for col in WEATHER_COLS:
            feature[col] = weather_vals[col]

        feature["hour"] = ts.hour
        feature["dow"] = ts.dayofweek
        feature["dayofyear"] = ts.dayofyear
        feature["month"] = ts.month
        feature["is_weekend"] = int(ts.dayofweek in (5, 6))

        row = pd.DataFrame([feature])[feature_cols].fillna(fill_values)
        pred = model.predict(row)[0]
        preds.append(
            {"timestamp_local": ts, "load_area": area_name, "prediction_mw": pred}
        )
        history_values.loc[ts] = pred
        history_flags.loc[ts] = True
        history_values = history_values.sort_index()
        history_flags = history_flags.sort_index()

    return pd.DataFrame(preds)


def train_and_predict(
    area_df: pd.DataFrame,
    feature_cols: list[str],
    input_df: pd.DataFrame,
    pred_start: pd.Timestamp | None = None,
    pred_end: pd.Timestamp | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    pred_start = pred_start or PRED_START
    pred_end = pred_end or PRED_END
    state = train_area_model(area_df, feature_cols, pred_start)
    return predict_area(
        state,
        feature_cols,
        input_df,
        pred_start=pred_start,
        pred_end=pred_end,
        verbose=verbose,
    )



def main():
    args = parse_args()
    load_dir = Path(args.load_dir)
    weather_csv = Path(args.weather_csv)

    load_df = load_load_history(load_dir)
    weather_df = load_weather(weather_csv)
    weather_lookup = weather_df.set_index(["load_area", "timestamp_local"])
    full_df = load_df.merge(
        weather_df, on=["load_area", "timestamp_local"], how="left", validate="m:1"
    )

    feature_cols = [f"lag_{lag}" for lag in LAG_HOURS] + [
        "hour",
        "dow",
        "dayofyear",
        "month",
        "is_weekend",
        *WEATHER_COLS,
    ]

    model_states = {}
    prediction_inputs = weather_df[
        (weather_df["timestamp_local"] >= PRED_START)
        & (weather_df["timestamp_local"] < PRED_END)
    ].copy()
    if prediction_inputs.empty:
        raise RuntimeError("No weather rows available for the default prediction window.")

    available_areas = set(prediction_inputs["load_area"].unique())
    predictions = []
    for load_area, area_df in full_df.groupby("load_area"):
        state = train_area_model(area_df, feature_cols, PRED_START)
        if not state:
            print(f"Skipping {load_area}: insufficient history")
            continue
        if load_area not in available_areas:
            print(f"Skipping {load_area}: no weather inputs in window")
            continue
        model_states[load_area] = state
        preds = predict_area(
            state,
            feature_cols,
            prediction_inputs,
            pred_start=PRED_START,
            pred_end=PRED_END,
            verbose=False,
        )
        if preds.empty:
            print(f"Skipping {load_area}: prediction window empty")
            continue
        predictions.append(preds)
        print(f"Predicted {len(preds)} rows for {load_area}")

    if not predictions:
        raise RuntimeError("No predictions generated.")

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df = pred_df.sort_values(["load_area", "timestamp_local"])
    pred_df["timestamp_local"] = pred_df["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    pred_df.to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
