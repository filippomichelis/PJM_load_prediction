PYTHON          ?= python3
PJM_DOWNLOAD    ?= pjm_download.py
WEATHER_FETCH   ?= fetch_weather_features.py
MAKE_DATASETS   ?= make_datasets.py
PREDICT_ARGS    ?=
PJM_ARGS        ?=
WEATHER_ARGS    ?= --out data/fetch_weather_features.csv
MAKE_DATASETS_ARGS ?=

.PHONY: predictions rawdata trainmodels clean

predictions:
	@$(PYTHON) predict_day.py $(PREDICT_ARGS)

predictions_training: rawdata trainmodels
	@$(PYTHON) predict_day.py $(PREDICT_ARGS)

rawdata:
	@echo "Clearing data directory..."
	@rm -rf data/*
	@echo "Updating PJM load history..."
	@$(PYTHON) $(PJM_DOWNLOAD) $(PJM_ARGS)
	@echo "Fetching weather features..."
	@$(PYTHON) $(WEATHER_FETCH) $(WEATHER_ARGS)
	@echo "Building combined datasets..."
	@$(PYTHON) $(MAKE_DATASETS) $(MAKE_DATASETS_ARGS)
	@$(PYTHON) november_peaks_summary.py

trainmodels:
	@echo "Training region models..."
	@$(PYTHON) train.py

clean:
	@echo "Removing derived artifacts..."
	@rm -rf artifacts weather_by_load_area __pycache__
	@echo "Pruning generated data files (keeping raw inputs)..."
	@-[ -d data ] && find data -mindepth 1 -maxdepth 1 ! -name raw -exec rm -rf {} + || true
	@echo "Removing compiled Python caches..."
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + >/dev/null 2>&1 || true
	@find . -name '*.pyc' -delete >/dev/null 2>&1 || true
	@rm -f predictions.csv
