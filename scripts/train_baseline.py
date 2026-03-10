from ml_training_practice.modeling import train_housing_univariate_linear_regression


def main() -> None:
    result = train_housing_univariate_linear_regression()
    print(f"Housing LinearRegression RMSE: {result.rmse:.2f}")
    print(f"Housing LinearRegression R^2: {result.r2:.3f}")


if __name__ == "__main__":
    main()
