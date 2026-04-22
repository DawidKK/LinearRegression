.PHONY: sync lint test train train-poly train-multi train-multi-scaled train-ridge-sweep train-lasso-sweep train-elastic-net-cv train-logistic-step1 notebook format

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix

test:
	uv run pytest -q

train:
	uv run python scripts/linear_regression/train_baseline.py

train-poly:
	uv run python scripts/linear_regression/train_housing_polynomial.py

train-multi:
	uv run python scripts/linear_regression/train_housing_multifeature_linear.py

train-multi-scaled:
	uv run python scripts/linear_regression/train_housing_multifeature_scaled_linear.py

notebook:
	uv run jupyter lab

train-ridge-sweep:
	uv run python scripts/linear_regression/train_housing_ridge_alpha_sweep.py

train-lasso-sweep:
	uv run python scripts/linear_regression/train_housing_lasso_alpha_sweep.py

train-elastic-net-cv:
	uv run python scripts/linear_regression/train_housing_elastic_net_cv.py

train-logistic-step1:
	uv run python scripts/logistic_regression/step1_raw_logistic_regression_baseline.py
