#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BEST_PARAMS: Dict[str, Dict[str, float | int]] = {
    "AECO":   {"learning_rate": 0.05, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "AEPAPT": {"learning_rate": 0.05, "max_depth": 12, "max_iter": 1000, "min_samples_leaf": 50},
    "AEPIMP": {"learning_rate": 0.03, "max_depth": 16, "max_iter": 400,  "min_samples_leaf": 50},
    "AEPKPT": {"learning_rate": 0.05, "max_depth": 12, "max_iter": 700,  "min_samples_leaf": 50},
    "AEPOPT": {"learning_rate": 0.03, "max_depth": 8,  "max_iter": 400,  "min_samples_leaf": 100},
    "AP":     {"learning_rate": 0.05, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "BC":     {"learning_rate": 0.05, "max_depth": 12, "max_iter": 700,  "min_samples_leaf": 50},
    "CE":     {"learning_rate": 0.03, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "DAY":    {"learning_rate": 0.05, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "DEOK":   {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "DOM":    {"learning_rate": 0.03, "max_depth": 8,  "max_iter": 400,  "min_samples_leaf": 50},
    "DPLCO":  {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "DUQ":    {"learning_rate": 0.03, "max_depth": 8, "max_iter": 700,  "min_samples_leaf": 50},
    "EASTON": {"learning_rate": 0.08, "max_depth": 16, "max_iter": 400,  "min_samples_leaf": 50},
    "EKPC":   {"learning_rate": 0.05, "max_depth": 8,  "max_iter": 700,  "min_samples_leaf": 50},
    "JC":     {"learning_rate": 0.05, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 100},
    "ME":     {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "OE":     {"learning_rate": 0.03, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "OVEC":   {"learning_rate": 0.05, "max_depth": 8, "max_iter": 700,  "min_samples_leaf": 50},
    "PAPWR":  {"learning_rate": 0.03, "max_depth": 8,  "max_iter": 400,  "min_samples_leaf": 50},
    "PE":     {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 100},
    "PEPCO":  {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "PLCO":   {"learning_rate": 0.05, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "PN":     {"learning_rate": 0.05, "max_depth": 8,  "max_iter": 700,  "min_samples_leaf": 50},
    "PS":     {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 50},
    "RECO":   {"learning_rate": 0.08, "max_depth": 16, "max_iter": 700,  "min_samples_leaf": 100},
    "SMECO":  {"learning_rate": 0.05, "max_depth": 12, "max_iter": 700,  "min_samples_leaf": 50},
    "UGI":    {"learning_rate": 0.03, "max_depth": 8,  "max_iter": 400,  "min_samples_leaf": 100},
    "VMEU":   {"learning_rate": 0.03, "max_depth": 8,  "max_iter": 700,  "min_samples_leaf": 50},
}
MISSING_THRESHOLD = 0.30
MIN_ROWS = 200
DEFAULT_TZ = "America/New_York"
TRAIN_START_DEFAULT = "2020-11-01 00:00:00"
TRAIN_END_DEFAULT = "2025-11-13 23:00:00"


@dataclass
class RegionModel:
    load_area: str
    pipeline: Pipeline
    kept_features: List[str]
    dropped_features: List[str]

    @property
    def model(self) -> HistGradientBoostingRegressor:
        return self.pipeline.named_steps["regressor"]


def _thanksgiving_dates(years: pd.Series, tz: str) -> set[pd.Timestamp]:
    dates = set()
    for year in map(int, years):
        date = pd.Timestamp(year=year, month=11, day=1, tz=tz)
        while date.weekday() != 3:
            date += pd.Timedelta(days=1)
        date += pd.Timedelta(days=21)
        dates.add(date.normalize())
    return dates


def _add_thanksgiving_flag(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["is_thanksgiving"] = pd.Series(dtype="int8")
        return df
    tg_dates = _thanksgiving_dates(df["timestamp_local"].dt.year.unique(), tz=tz)
    df["is_thanksgiving"] = df["timestamp_local"].dt.normalize().isin(tg_dates).astype("int8")
    return df


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No features left after dropping high-NaN columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_models_from_full_datasets(
    datasets: Dict[str, Tuple[pd.DataFrame, List[str]]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz_name: str,
    min_rows: int = MIN_ROWS,
    missing_threshold: float = MISSING_THRESHOLD,
) -> List[RegionModel]:
    region_models: List[RegionModel] = []
    for load_area, (area_df, base_feature_cols) in datasets.items():
        params = BEST_PARAMS.get(
            load_area,
            {"learning_rate": 0.05, "max_depth": 16, "max_iter": 500, "min_samples_leaf": 50},
        )
        window = area_df[
            (area_df["timestamp_local"] >= start) & (area_df["timestamp_local"] < end)
        ].copy()
        window = window.dropna(subset=["mw"])
        if len(window) < min_rows:
            continue

        window = _add_thanksgiving_flag(window, tz=tz_name)
        candidate_cols = [col for col in base_feature_cols if col != "dayofyear"]
        if "is_thanksgiving" not in candidate_cols:
            candidate_cols.append("is_thanksgiving")

        missing_ratio = window[candidate_cols].isna().mean()
        drop_cols = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        keep_cols = [col for col in candidate_cols if col not in drop_cols]

        categorical_cols = [col for col in ("month", "hour") if col in keep_cols]
        numeric_cols = [col for col in keep_cols if col not in categorical_cols]
        if not keep_cols or (not numeric_cols and not categorical_cols):
            continue

        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        max_iter=params["max_iter"],
                        min_samples_leaf=params["min_samples_leaf"],
                        random_state=0,
                    ),
                ),
            ]
        )
        pipeline.fit(window[keep_cols], window["mw"])
        region_models.append(
            RegionModel(
                load_area=load_area,
                pipeline=pipeline,
                kept_features=keep_cols,
                dropped_features=drop_cols,
            )
        )
    return region_models


def parse_timestamp(value: str, tz_name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_name)
    else:
        ts = ts.tz_convert(tz_name)
    return ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-load-area HistGradientBoosting models.")
    parser.add_argument(
        "--full-datasets",
        type=Path,
        default=Path("data/full_datasets.joblib"),
        help="Path to the prebuilt full_datasets joblib.",
    )
    parser.add_argument("--tz", type=str, default=DEFAULT_TZ, help="IANA timezone for timestamps.")
    parser.add_argument("--train-start", type=str, default=TRAIN_START_DEFAULT)
    parser.add_argument("--train-end", type=str, default=TRAIN_END_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--dont-drop-lag24",
        action="store_true",
        help="Remove lag_24 feature before training (expects column to exist in full_datasets).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = joblib.load(args.full_datasets)

    if not args.dont_drop_lag24:
        for load_area, (area_df, feature_cols) in list(datasets.items()):
            if "lag_24" in area_df.columns:
                area_df = area_df.drop(columns=["lag_24"])
            feature_cols = [col for col in feature_cols if col != "lag_24"]
            datasets[load_area] = (area_df, feature_cols)

    train_start = parse_timestamp(args.train_start, args.tz)
    train_end = parse_timestamp(args.train_end, args.tz)

    region_models = train_models_from_full_datasets(datasets, train_start, train_end, args.tz)
    if not region_models:
        raise RuntimeError("No region models were trained.")
    print(f"Trained {len(region_models)} region pipelines.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    region_payload = {
        rm.load_area: {
            "pipeline": rm.pipeline,
            "kept_features": rm.kept_features,
            "dropped_features": rm.dropped_features,
            "hyperparams": BEST_PARAMS.get(rm.load_area, {}),
            "train_start": str(train_start),
            "train_end": str(train_end),
        }
        for rm in region_models
    }
    joblib.dump(region_payload, out_dir / "region_models.joblib")


    # Terminal commands
    metadata = {
        "train_start": str(train_start),
        "train_end": str(train_end),
        "tz_name": args.tz,
        "full_datasets_path": str(Path(args.full_datasets).resolve()),
        "drop_lag24": not args.dont_drop_lag24,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Artifacts saved under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
