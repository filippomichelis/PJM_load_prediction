#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from evaluate_window import RegionModelPayload, evaluate_region_models

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"
FULL_DATASETS_PATH = BASE_DIR / "data/full_datasets.joblib"
LOAD_2025_PATH = BASE_DIR / "data/hrl_load_metered_2025.csv"
PROJECTION_ARTIFACT = MODEL_DIR / "projection_matrix.joblib"

TZ_NAME = "America/New_York"
DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_N_PCS = 4


def _ensure_timestamp(value: pd.Timestamp | str, tz_name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _load_region_models(model_dir: Path) -> List[RegionModelPayload]:
    model_path = Path(model_dir) / "region_models.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Region model artifact not found: {model_path}")
    payload = joblib.load(model_path)
    return [
        RegionModelPayload(
            load_area=area,
            pipeline=blob["pipeline"],
            kept_features=blob["kept_features"],
        )
        for area, blob in payload.items()
    ]


def _load_full_datasets(path: Path) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    full_path = Path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Full dataset cache missing: {full_path}")
    return joblib.load(full_path)


def find_recent_data_window(
    load_csv: Path,
    tz_name: str = TZ_NAME,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    reference_timestamp: pd.Timestamp | str | None = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    load_path = Path(load_csv)
    if not load_path.exists():
        raise FileNotFoundError(f"Load history file missing: {load_path}")

    frame = pd.read_csv(load_path, usecols=["datetime_beginning_ept"])
    frame["datetime_beginning_ept"] = pd.to_datetime(
        frame["datetime_beginning_ept"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    frame = frame.dropna(subset=["datetime_beginning_ept"])
    timestamps = frame["datetime_beginning_ept"].dt.tz_localize(
        tz_name, nonexistent="shift_forward", ambiguous="NaT"
    )
    timestamps = timestamps.dropna()
    if reference_timestamp is None:
        reference_ts = pd.Timestamp.now(tz=tz_name)
    else:
        reference_ts = pd.Timestamp(reference_timestamp)
        if reference_ts.tz is None:
            reference_ts = reference_ts.tz_localize(tz_name)
        else:
            reference_ts = reference_ts.tz_convert(tz_name)

    filtered_ts = timestamps[timestamps < reference_ts]
    if filtered_ts.empty:
        # Fall back to the latest available data if reference is earlier than history.
        filtered_ts = timestamps
    if filtered_ts.empty:
        raise RuntimeError("No timestamps found in the provided load history file.")

    daily_index = filtered_ts.dt.normalize().dropna().drop_duplicates().sort_values()
    cutoff_day = (reference_ts - pd.Timedelta(microseconds=1)).normalize()
    recent_days = daily_index[daily_index <= cutoff_day]
    if recent_days.empty:
        recent_days = daily_index
    if recent_days.empty:
        raise RuntimeError("No completed days found in recent history.")
    if len(recent_days) < lookback_days:
        raise RuntimeError(
            f"Only {len(recent_days)} days available, need {lookback_days} to build projection matrix."
        )
    selected = recent_days.iloc[-lookback_days:]
    window_start = selected.iloc[0]
    window_end = selected.iloc[-1] + pd.Timedelta(days=1)
    return window_start, window_end


def build_projection_features(
    preds_df: pd.DataFrame,
    n_pcs: int = DEFAULT_N_PCS,
    drop_missing_targets: bool = False,
) -> pd.DataFrame:
    if preds_df.empty:
        return pd.DataFrame()

    df = preds_df.copy()
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.dropna(subset=["timestamp_local", "load_area", "mw_pred"])
    df["load_area"] = df["load_area"].astype(str)
    df = df.sort_values(["timestamp_local", "load_area"])

    pred_matrix = df.pivot_table(
        index="timestamp_local",
        columns="load_area",
        values="mw_pred",
    ).sort_index()
    if pred_matrix.empty:
        return pd.DataFrame()

    filled_matrix = pred_matrix.fillna(0.0)
    pc_cols = [f"pred_pc{i+1}" for i in range(max(0, n_pcs))]
    pc_df = pd.DataFrame(0.0, index=pred_matrix.index, columns=pc_cols)
    n_components = min(len(pc_cols), filled_matrix.shape[0], filled_matrix.shape[1])
    if n_components > 0:
        pcs = PCA(n_components=n_components, random_state=0).fit_transform(filled_matrix.values)
        for idx in range(n_components):
            pc_df.iloc[:, idx] = pcs[:, idx]

    preds_lookup = df.set_index(["timestamp_local", "load_area"])
    rows = []
    for ts in pred_matrix.index:
        for area in pred_matrix.columns:
            key = (ts, area)
            if key not in preds_lookup.index:
                continue
            record = preds_lookup.loc[key]
            if isinstance(record, pd.DataFrame):
                record = record.iloc[0]
            row = {
                "load_area": area,
                "timestamp_local": ts,
                "base_pred": float(record["mw_pred"]),
            }
            actual = record.get("mw_actual", np.nan)
            row["mw_actual"] = actual
            if not pd.isna(actual):
                row["target_residual"] = float(actual) - row["base_pred"]
            for col in pc_cols:
                row[col] = pc_df.loc[ts, col] if col in pc_df.columns else 0.0
            rows.append(row)

    stack_df = pd.DataFrame(rows)
    if stack_df.empty:
        return stack_df

    if "target_residual" not in stack_df.columns:
        stack_df["target_residual"] = np.nan
    if drop_missing_targets:
        stack_df = stack_df.dropna(subset=["target_residual"])
    for col in pc_cols:
        if col not in stack_df.columns:
            stack_df[col] = 0.0
    return stack_df.reset_index(drop=True)


def train_recent_projection_matrix(
    model_dir: Path = MODEL_DIR,
    full_datasets_path: Path = FULL_DATASETS_PATH,
    load_csv: Path = LOAD_2025_PATH,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    n_pcs: int = DEFAULT_N_PCS,
    tz_name: str = TZ_NAME,
    artifact_path: Path = PROJECTION_ARTIFACT,
    reference_timestamp: pd.Timestamp | str | None = None,
) -> Dict[str, object]:
    window_start, window_end = find_recent_data_window(
        load_csv,
        tz_name=tz_name,
        lookback_days=lookback_days,
        reference_timestamp=reference_timestamp,
    )
    metadata = {
        "lookback_days": lookback_days,
        "reference_timestamp": reference_timestamp,
    }
    return train_projection_for_window(
        window_start=window_start,
        window_end=window_end,
        model_dir=model_dir,
        full_datasets_path=full_datasets_path,
        n_pcs=n_pcs,
        tz_name=tz_name,
        artifact_path=artifact_path,
        extra_metadata=metadata,
    )


def train_projection_for_window(
    window_start: pd.Timestamp | str,
    window_end: pd.Timestamp | str,
    model_dir: Path = MODEL_DIR,
    full_datasets_path: Path = FULL_DATASETS_PATH,
    n_pcs: int = DEFAULT_N_PCS,
    tz_name: str = TZ_NAME,
    artifact_path: Path = PROJECTION_ARTIFACT,
    extra_metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Fit and persist a projection matrix using predictions from a fixed evaluation window."""
    start_ts = _ensure_timestamp(window_start, tz_name)
    end_ts = _ensure_timestamp(window_end, tz_name)
    if end_ts <= start_ts:
        raise ValueError("window_end must be greater than window_start.")

    region_models = _load_region_models(model_dir)
    if not region_models:
        raise RuntimeError("No region models available to evaluate.")
    full_datasets = _load_full_datasets(full_datasets_path)

    preds_df, _, _ = evaluate_region_models(
        region_models,
        full_datasets,
        test_start=start_ts,
        test_end=end_ts,
        tz_name=tz_name,
    )
    stack_df = build_projection_features(preds_df, n_pcs=n_pcs, drop_missing_targets=True)
    if stack_df.empty:
        raise RuntimeError("Insufficient predictions to fit projection matrix.")

    pc_cols = [col for col in stack_df.columns if col.startswith("pred_pc")]
    feature_cols = ["base_pred", *pc_cols]
    for col in feature_cols:
        if col not in stack_df.columns:
            stack_df[col] = 0.0

    projection_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=400,
                    random_state=0,
                ),
            ),
        ]
    )
    projection_pipeline.fit(stack_df[feature_cols], stack_df["target_residual"])

    stack_df["residual_correction"] = projection_pipeline.predict(stack_df[feature_cols])
    stack_df["mw_pred_corrected"] = stack_df["base_pred"] + stack_df["residual_correction"]
    base_mse = mean_squared_error(stack_df["mw_actual"], stack_df["base_pred"])
    corrected_mse = mean_squared_error(stack_df["mw_actual"], stack_df["mw_pred_corrected"])

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "pipeline": projection_pipeline,
        "feature_columns": feature_cols,
        "n_pcs": len(pc_cols),
        "window_start": start_ts,
        "window_end": end_ts,
        "tz_name": tz_name,
    }
    if extra_metadata:
        artifact_payload.update(extra_metadata)
    joblib.dump(artifact_payload, artifact_path)

    return {
        "artifact_path": artifact_path,
        "window_start": start_ts,
        "window_end": end_ts,
        "base_mse": float(base_mse),
        "corrected_mse": float(corrected_mse),
    }


def apply_projection_correction(
    preds_df: pd.DataFrame,
    artifact_path: Path = PROJECTION_ARTIFACT,
    n_pcs: int | None = None,
    apply_mask: pd.Series | List[bool] | None = None,
) -> pd.DataFrame:
    """Apply saved projection corrections to a prediction DataFrame.

    Parameters
    ----------
    preds_df
        DataFrame containing at minimum ``load_area``, ``timestamp_local`` and ``mw_pred``.
    artifact_path
        Path to the saved projection artifact produced by ``train_recent_projection_matrix``.
    n_pcs
        Overrides the number of principal components used when rebuilding projection features.
    apply_mask
        Optional boolean mask (aligned to ``preds_df``) indicating which rows were produced
        by ML models and should therefore receive projection corrections. Rows with mask
        ``False`` will be passed through unchanged.
    """

    if preds_df.empty:
        return preds_df
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Projection matrix artifact missing: {artifact_path}")

    stacker_meta = joblib.load(artifact_path)
    feature_cols = stacker_meta["feature_columns"]
    n_pcs = n_pcs or stacker_meta.get("n_pcs", len([c for c in feature_cols if c.startswith("pred_pc")]))

    df = preds_df.copy()
    df["_row_id"] = np.arange(len(df))
    df["mw_pred_corrected"] = df["mw_pred"]

    if apply_mask is not None:
        mask_series = pd.Series(apply_mask)
        if mask_series.index.equals(df.index):
            selection = mask_series.astype(bool)
        elif len(mask_series) == len(df):
            selection = pd.Series(mask_series.values, index=df.index).astype(bool)
        else:
            selection = mask_series.reindex(df.index).fillna(False).astype(bool)
    else:
        selection = pd.Series(True, index=df.index)

    target_df = df[selection].copy()
    if target_df.empty:
        df["residual_correction"] = 0.0
        return df.drop(columns="_row_id")

    stack_features = build_projection_features(target_df, n_pcs=n_pcs, drop_missing_targets=False)
    if stack_features.empty:
        df["residual_correction"] = 0.0
        return df.drop(columns="_row_id")

    for col in feature_cols:
        if col not in stack_features.columns:
            stack_features[col] = 0.0

    corrections = stacker_meta["pipeline"].predict(stack_features[feature_cols])
    stack_features = stack_features.assign(residual_correction=corrections)
    corrections_df = stack_features[["load_area", "timestamp_local", "residual_correction"]]

    df = df.merge(
        corrections_df,
        on=["load_area", "timestamp_local"],
        how="left",
    )
    df = df.sort_values("_row_id").drop(columns="_row_id")
    df["residual_correction"] = df["residual_correction"].fillna(0.0)
    df["mw_pred_corrected"] = df["mw_pred"] + df["residual_correction"]
    return df


if __name__ == "__main__":
    summary = train_recent_projection_matrix()
    print(
        "Projection matrix trained for "
        f"{summary['window_start']} to {summary['window_end']} and stored at {summary['artifact_path']}"
    )
