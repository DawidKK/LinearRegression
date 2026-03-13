import argparse

import matplotlib.pyplot as plt

from ml_training_practice.modeling import train_housing_univariate_polynomial_regression
from ml_training_practice.visualization import plot_housing_polynomial_regression_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train polynomial regression models for housing prices and compare degrees."
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=1,
        help="Minimum polynomial degree (inclusive).",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=5,
        help="Maximum polynomial degree (inclusive).",
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/housing_polynomial_comparison.png",
        help="Where to save the generated comparison plot.",
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


def print_results_table() -> None:
    print("\nDegree comparison on test split")
    print("-" * 40)
    print(f"{'Degree':<10}{'RMSE':>14}{'R^2':>12}")
    print("-" * 40)


def main() -> None:
    args = parse_args()
    results = train_housing_univariate_polynomial_regression(
        min_degree=args.min_degree,
        max_degree=args.max_degree,
    )

    print_results_table()
    for result in results:
        print(f"{result.degree:<10}{result.rmse:>14.2f}{result.r2:>12.3f}")
    print("-" * 40)

    save_path = None if args.no_save else args.output_path
    _, _ = plot_housing_polynomial_regression_comparison(results, save_path=save_path)

    if save_path:
        print(f"Saved figure to: {save_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
