from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from .data import load_housing_univariate_feature_target
from .preprocessing import split_regression_data


@dataclass
class HousingLinearRegressionResult:
    # Single object to keep everything needed for evaluation + plotting + interpretation.
    model: LinearRegression
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_predictions: pd.Series
    rmse: float
    r2: float
    coefficient: float
    intercept: float


def train_housing_univariate_linear_regression(
    test_size: float = 0.2,
    random_state: int = 42,
    csv_path: str = "data/raw/housing_prices_sample.csv",
) -> HousingLinearRegressionResult:
    # Use one feature only: year_built -> price_usd.
    features, target = load_housing_univariate_feature_target(csv_path=csv_path)

    # Split first, then fit only on train data to avoid evaluation leakage.
    x_train, x_test, y_train, y_test = split_regression_data(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    # Keep test prediction index aligned with y_test for easier inspection/comparison.
    test_predictions = pd.Series(model.predict(x_test), index=y_test.index)
    # Regression metrics requested for this learning workflow.
    rmse = float(root_mean_squared_error(y_test, test_predictions))
    r2 = float(r2_score(y_test, test_predictions))

    return HousingLinearRegressionResult(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        test_predictions=test_predictions,
        rmse=rmse,
        r2=r2,
        coefficient=float(model.coef_[0]),
        intercept=float(model.intercept_),
    )
