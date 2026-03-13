from pathlib import Path

import pandas as pd


def load_housing_prices_data(csv_path: str | Path = "data/raw/housing_prices_sample.csv") -> pd.DataFrame:
    # Build an absolute path from the project root so this works from any working directory.
    project_root = Path(__file__).resolve().parents[2]
    resolved_csv_path = (project_root / Path(csv_path)).resolve()
    return pd.read_csv(resolved_csv_path)


def load_housing_univariate_feature_target(
    feature_name: str = "year_built",
    target_name: str = "price_usd",
    csv_path: str | Path = "data/raw/housing_prices_sample.csv",
) -> tuple[pd.DataFrame, pd.Series]:
    housing_data = load_housing_prices_data(csv_path=csv_path)
    # Double brackets keep features as a DataFrame (shape: n_samples x 1), which sklearn expects.
    features = housing_data[[feature_name]]
    # Target is a Series (shape: n_samples).
    target = housing_data[target_name]
    return features, target
