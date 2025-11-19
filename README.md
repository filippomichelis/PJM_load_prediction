### PJM Load Prediction Toolkit

This project contains everything needed to rebuild load datasets, train per-load-area models, and generate autoregressive daily forecasts for the PJM electricity load by 29 areas of disaggregation.
The main model builds on 29 HistGradientBoosting models trained individually for each region. The final output are projected on the PCA components coming from the residual of the most recent 5 days of load data.

The models use lagged loads as well as weather forecasts for the 5 main cities within each region of disaggregation.

#### Repository layout

- `Makefile` – defines the main targets (`predictions`, `rawdata`, `trainmodels`, `clean`).
- `requirements.txt` – pinned versions for pandas, scikit-learn, Meteostat, etc.
- `predict_day.py` – autoregressive inference script with `--verbose` logging, projection correction, and peak clipping.
- `train.py` – trains HistGradientBoosting models for each PJM load area and writes `artifacts/region_models.joblib`.
- `make_datasets.py` – builds `data/full_datasets.joblib` by merging PJM load history with weather features.
- `pjm_download.py`, `fetch_weather_features.py` – fetch PJM hourly load (API/OSF) and Meteostat weather snapshots.
- `projection_stacker.py` – creates/applies PCA-based residual correction.
- `november_peaks_summary.py` – summarizes November min/max values per area (used to clamp predictions).
- `Dockerfile` – wraps the entire project so every Make target works inside a container.

Artifacts (`region_models.joblib`, `projection_matrix.joblib`, `november_peak_summary.joblib`, etc.) live under `docker/artifacts/`. Raw and derived data are written to `docker/data/`.

#### Quickstart

```bash
cd docker
python -m venv .venv
.\.venv\Scripts\activate              # or source .venv/bin/activate on Unix
pip install --upgrade pip
pip install -r requirements.txt
```

Run the full pipeline (fetch latest inputs, retrain, predict tomorrow) with:

```bash
make predictions
make predictions PREDICT_ARGS="--verbose"   # include extra logging
```

Other useful targets:

- `make rawdata` – clears `data/` (except `data/raw`) and reruns `pjm_download.py`, `fetch_weather_features.py`, `make_datasets.py`.
- `make trainmodels` – rebuilds all region models via `train.py`.
- `make clean` – removes artifacts, generated data, caches (leaves raw inputs intact).

#### Docker usage

```bash
cd docker
docker build -t filippomichelis/load_prediction .
docker run -it --rm filippomichelis/load_prediction make predictions
docker run -it --rm filippomichelis/load_prediction make predictions PREDICT_ARGS="--verbose"
```

The Dockerfile installs requirements and copies the entire project into `/home/jovyan/app`, so the Makefile works unchanged inside the container.

#### Supporting scripts

- `projection_stacker.py` automatically retrains the residual correction window; run manually if needed.
- `november_peaks_summary.py` regenerates the min/max November lookup (saved under `artifacts/november_peak_summary.joblib`).
- `autoregressive_projection.py` demonstrates the full prediction flow outside of Make.

#### Data expectations

- `data/hrl_load_metered_YYYY.csv` – hourly PJM load (via `pjm_download.py`).
- `data/weather_points_oct_nov_dec.csv` – Meteostat aggregates (via `fetch_weather_features.py`).
- `weather_by_load_area/` – per-area weather matrices (auto-created by `make_datasets.py`).
- `data/full_datasets.joblib` – cached feature bundle consumed by `train.py` and `predict_day.py`.

#### Notes

- `predict_day.py` feeds corrected predictions back into lag features to forecast multiple days ahead. It also clamps residual-corrected outputs within the November ranges stored in `artifacts/november_peak_summary.joblib`.
- `make predictions` already depends on `rawdata` and `trainmodels`, so it refreshes inputs and retrains models before forecasting.
- Pass `--verbose` to any supported script (via `PREDICT_ARGS`, or directly) if you want more runtime detail.
