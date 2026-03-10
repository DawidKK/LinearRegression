.PHONY: sync lint test train notebook format

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

notebook:
	uv run jupyter lab
