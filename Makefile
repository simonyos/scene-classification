.PHONY: setup features train-tabular train-cnn eval serve test lint fmt docker clean

PY ?= python
UV ?= uv

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

features:
	$(PY) -m scene_classification.cli extract-features

train-tabular:
	$(PY) -m scene_classification.cli train-tabular

train-cnn:
	$(PY) -m scene_classification.cli train-cnn

eval:
	$(PY) -m scene_classification.cli evaluate

serve:
	uvicorn scene_classification.serve.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

docker:
	docker build -t scene-classification:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov mlruns artifacts
