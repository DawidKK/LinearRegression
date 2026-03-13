# ML Training Practice (uv)

Package-first machine learning training repository using `uv`.

## Quick start

1. Sync dependencies:

   ```bash
   uv sync
   ```

2. Run tests:

   ```bash
   uv run pytest
   ```

3. Run lint:

   ```bash
   uv run ruff check .
   ```

4. Start JupyterLab:

   ```bash
   uv run jupyter lab
   ```

## Project structure

```
src/ml_training_practice/  # Reusable training code
tests/                     # Unit and smoke tests
notebooks/                 # Experiment notebooks
scripts/                   # Small executable scripts
data/raw/                  # Local raw datasets (gitignored)
data/processed/            # Local processed datasets (gitignored)
models/                    # Saved model artifacts (gitignored)
reports/figures/           # Generated figures (gitignored)
```

## Typical workflow

- Build reusable logic in `src/ml_training_practice`.
- Use notebooks for experiments, but import package code instead of duplicating logic.
- Keep large/local artifacts in `data/`, `models/`, and `reports/figures/`.

## Make shortcuts

Run common tasks with short commands:

```bash
make sync
make lint
make test
make train
make notebook
```

## Polynomial Regression Practice

Run the polynomial degree comparison (default: degrees 1..5):

```bash
uv run python scripts/train_housing_polynomial.py
```

Or via Make:

```bash
make train-poly
```
