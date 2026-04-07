import argparse

import matplotlib.pyplot as plt

from ml_training_practice.modeling import train_housing_univariate_linear_regression
from ml_training_practice.visualization import plot_housing_linear_regression_result


def parse_args() -> argparse.Namespace:
    # Simple CLI flags: where to save, whether to save, and whether to display.
    parser = argparse.ArgumentParser(
        description="Train a pure linear regression model and plot housing price curve."
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/housing_linear_curve.png",
        help="Where to save the generated plot.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the plot to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot in an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Runs pure linear regression (no regularization, no scaling, one feature).
    result = train_housing_univariate_linear_regression()
    save_path = None if args.no_save else args.output_path
    # Creates the chart and optionally writes it to disk.
    _, _ = plot_housing_linear_regression_result(result, save_path=save_path)

    # Print core evaluation and interpretation numbers in terminal.
    print(f"RMSE: {result.rmse:.2f}")
    print(f"R^2: {result.r2:.3f}")
    print(f"Coefficient (year_built): {result.coefficient:.2f}")
    print(f"Intercept: {result.intercept:.2f}")

    if save_path:
        print(f"Saved figure to: {save_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
