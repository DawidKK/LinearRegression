.PHONY: sync lint test train train-poly train-multi notebook format

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

notebook:
	uv run jupyter lab
