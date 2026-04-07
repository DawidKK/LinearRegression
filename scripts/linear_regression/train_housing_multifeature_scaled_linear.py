import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from ml_training_practice.data import load_housing_prices_data
from ml_training_practice.preprocessing import split_regression_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multifeature linear regression with StandardScaler on housing data."
    )
    parser.add_argument(
        "--target",
        default="price_usd",
        help="Target column name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to use for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/housing_multifeature_scaled_linear_regression.png",
        help="Where to save the result figure (.png).",
    )
    parser.add_argument(
        "--csv-path",
        default="data/raw/housing_prices_sample.csv",
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated plot in an interactive window.",
    )
    return parser.parse_args()


def build_feature_target_frames(data: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_name not in data.columns:
        raise ValueError(f"Target column '{target_name}' is not in dataset.")

    # Use every column except the target as model input features.
    feature_columns = [column for column in data.columns if column != target_name]
    features = data[feature_columns]
    target = data[target_name]
    return features, target


def plot_results(
    y_test: pd.Series,
    predictions: pd.Series,
    feature_names: list[str],
    scaled_coefficients: list[float],
    rmse: float,
    r2: float,
    save_path: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left chart: ideal line vs predicted values for quick error inspection.
    axes[0].scatter(y_test, predictions, alpha=0.8, color="#1f77b4")
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    axes[0].plot([min_value, max_value], [min_value, max_value], "r--", linewidth=2)
    axes[0].set_title("Actual vs Predicted Prices (Scaled Features)")
    axes[0].set_xlabel("Actual price_usd")
    axes[0].set_ylabel("Predicted price_usd")
    axes[0].grid(alpha=0.25)

    metrics_text = f"RMSE: {rmse:,.2f}\nR^2: {r2:.3f}"
    axes[0].text(
        0.02,
        0.98,
        metrics_text,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    # Right chart: coefficients in scaled feature space for easier magnitude comparison.
    coefficient_frame = pd.DataFrame(
        {"feature": feature_names, "coefficient": scaled_coefficients}
    ).sort_values("coefficient")

    axes[1].barh(coefficient_frame["feature"], coefficient_frame["coefficient"], color="#2ca02c")
    axes[1].set_title("Learned Coefficients (Scaled Features)")
    axes[1].set_xlabel("Coefficient value")
    axes[1].grid(axis="x", alpha=0.25)

    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)


def main() -> None:
    args = parse_args()

    housing_data = load_housing_prices_data(csv_path=args.csv_path)
    features, target = build_feature_target_frames(housing_data, target_name=args.target)

    x_train, x_test, y_train, y_test = split_regression_data(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Fit scaler only on training data to avoid leakage into evaluation data.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    predictions = pd.Series(model.predict(x_test_scaled), index=y_test.index)
    rmse = float(root_mean_squared_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    print("Multifeature Housing Linear Regression (with StandardScaler)")
    print("-" * 62)
    print(f"Features used: {x_train.shape[1]}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.3f}")
    print(f"Intercept: {model.intercept_:.2f}")

    print("\nCoefficients (scaled feature space):")
    for feature_name, coefficient in zip(x_train.columns, model.coef_):
        print(f"{feature_name:<24}{coefficient:>12.4f}")

    plot_results(
        y_test=y_test,
        predictions=predictions,
        feature_names=list(x_train.columns),
        scaled_coefficients=list(model.coef_),
        rmse=rmse,
        r2=r2,
        save_path=args.output_path,
    )
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
