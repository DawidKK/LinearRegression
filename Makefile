.PHONY: sync lint test train train-poly train-multi train-multi-scaled train-ridge-sweep train-lasso-sweep notebook format

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix

test:
	uv run pytest -q

train:
	uv run python scripts/train_baseline.py

train-poly:
	uv run python scripts/train_housing_polynomial.py

train-multi:
	uv run python scripts/train_housing_multifeature_linear.py

train-multi-scaled:
	uv run python scripts/train_housing_multifeature_scaled_linear.py

notebook:
	uv run jupyter lab

train-ridge-sweep:
	uv run python scripts/train_housing_ridge_alpha_sweep.py

train-lasso-sweep:
	uv run python scripts/train_housing_lasso_alpha_sweep.py
