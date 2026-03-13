.PHONY: sync lint test train train-poly notebook format

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

notebook:
	uv run jupyter lab
