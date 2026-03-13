from .modeling import (
    train_housing_univariate_linear_regression,
    train_housing_univariate_polynomial_regression,
)


def main() -> None:
    result = train_housing_univariate_linear_regression()
    print(f"Trained housing LinearRegression model. RMSE={result.rmse:.2f}, R^2={result.r2:.3f}")


__all__ = [
    "main",
    "train_housing_univariate_linear_regression",
    "train_housing_univariate_polynomial_regression",
]
