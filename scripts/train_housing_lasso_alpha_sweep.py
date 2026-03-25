import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from ml_training_practice.data import load_housing_prices_data
from ml_training_practice.preprocessing import split_regression_data


# This script runs a simple Lasso (L1) experiment for multiple alpha values.
# Why this is useful:
# - Alpha controls regularization strength.
# - Larger alpha pushes more coefficients toward exactly 0.
# - Seeing metrics + coefficient behavior together builds intuition.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Lasso alpha sweep on housing data and save a comparison figure."
    )
    parser.add_argument(
        "--target",
        default="price_usd",
        help="Target column name.",
    )
    parser.add_argument(
        "--alphas",
        default="0.001,0.01,0.1,1,10,100",
        help="Comma-separated alpha values, for example: 0.001,0.01,0.1,1,10,100",
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
        default="reports/figures/housing_lasso_alpha_sweep.png",
        help="Where to save the comparison figure (.png).",
    )
    parser.add_argument(
        "--csv-path",
        default="data/raw/housing_prices_sample.csv",
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Maximum optimization iterations for Lasso solver.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated plot in an interactive window.",
    )
    return parser.parse_args()


def parse_alpha_values(alpha_text: str) -> list[float]:
    # Parse user text like "0.001,0.01,0.1" into a list of floats.
    parts = [part.strip() for part in alpha_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one alpha value must be provided.")

    alphas = [float(part) for part in parts]
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("All alpha values must be greater than 0.")
    return alphas


def build_feature_target_frames(data: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    # Keep all columns except target as features.
    if target_name not in data.columns:
        raise ValueError(f"Target column '{target_name}' is not in dataset.")

    feature_columns = [column for column in data.columns if column != target_name]
    features = data[feature_columns]
    target = data[target_name]
    return features, target


def plot_sweep_results(results: pd.DataFrame, save_path: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Metric view: model quality vs alpha.
    axes[0].plot(results["alpha"], results["rmse"], marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_xscale("log")
    axes[0].set_title("RMSE vs Alpha (Lasso)")
    axes[0].set_xlabel("Alpha (log scale)")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)

    # Sparsity view: how many coefficients are exactly 0.
    axes[1].plot(results["alpha"], results["zero_coefficients"], marker="o", linewidth=2, color="#d62728")
    axes[1].set_xscale("log")
    axes[1].set_title("Zero Coefficients vs Alpha")
    axes[1].set_xlabel("Alpha (log scale)")
    axes[1].set_ylabel("Number of coefficients equal to 0")
    axes[1].grid(alpha=0.25)

    # Magnitude view: total coefficient size (L1 norm) vs alpha.
    axes[2].plot(results["alpha"], results["coef_l1_norm"], marker="o", linewidth=2, color="#2ca02c")
    axes[2].set_xscale("log")
    axes[2].set_title("Coefficient L1 Norm vs Alpha")
    axes[2].set_xlabel("Alpha (log scale)")
    axes[2].set_ylabel("L1 norm of coefficients")
    axes[2].grid(alpha=0.25)

    figure.suptitle(
        f"Lasso Alpha Sweep (best R^2 = {results['r2'].max():.3f})",
        fontsize=12,
    )
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)


def main() -> None:
    args = parse_args()
    alphas = parse_alpha_values(args.alphas)

    # 1) Load tabular data.
    housing_data = load_housing_prices_data(csv_path=args.csv_path)

    # 2) Separate feature matrix X and target vector y.
    features, target = build_feature_target_frames(housing_data, target_name=args.target)

    # 3) Train/test split for honest evaluation.
    x_train, x_test, y_train, y_test = split_regression_data(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 4) Scale inputs (important for Lasso: regularization is scale-sensitive).
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 5) Train one model per alpha and collect comparison stats.
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=args.max_iter)
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)

        # Print feature coefficients for this alpha so we can directly observe
        # which features shrink and which become exactly 0 as regularization grows.
        coefficient_series = pd.Series(model.coef_, index=features.columns)
        print(f"\nCoefficients for alpha={alpha:.4g}")
        print(coefficient_series)

        rmse = float(root_mean_squared_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        # Lasso-specific introspection:
        # - coef_l1_norm: total absolute magnitude of coefficients.
        # - zero_coefficients: how many features were effectively removed.
        coef_l1_norm = float(np.linalg.norm(model.coef_, ord=1))
        zero_coefficients = float(np.sum(np.isclose(model.coef_, 0.0)))

        rows.append(
            {
                "alpha": alpha,
                "rmse": rmse,
                "r2": r2,
                "coef_l1_norm": coef_l1_norm,
                "zero_coefficients": zero_coefficients,
            }
        )

    results = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

    print("Housing Lasso Alpha Sweep")
    print("-" * 66)
    print(f"{'Alpha':>10}{'RMSE':>14}{'R^2':>12}{'Coef L1 Norm':>16}{'Zero Coefs':>14}")
    print("-" * 66)
    for _, row in results.iterrows():
        print(
            f"{row['alpha']:>10.4g}{row['rmse']:>14.2f}{row['r2']:>12.3f}"
            f"{row['coef_l1_norm']:>16.2f}{row['zero_coefficients']:>14.0f}"
        )
    print("-" * 66)

    best_row = results.loc[results["r2"].idxmax()]
    print(
        f"Best by R^2: alpha={best_row['alpha']:.4g}, "
        f"RMSE={best_row['rmse']:.2f}, R^2={best_row['r2']:.3f}, "
        f"Zero coefs={best_row['zero_coefficients']:.0f}"
    )

    plot_sweep_results(results=results, save_path=args.output_path)
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
