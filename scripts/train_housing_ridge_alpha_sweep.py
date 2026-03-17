import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from ml_training_practice.data import load_housing_prices_data
from ml_training_practice.preprocessing import split_regression_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small Ridge alpha sweep on housing data and save comparison plots."
    )
    parser.add_argument(
        "--target",
        default="price_usd",
        help="Target column name.",
    )
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
        help="Comma-separated alpha values, for example: 0.01,0.1,1,10,100",
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
        default="reports/figures/housing_ridge_alpha_sweep.png",
        help="Where to save the comparison figure (.png).",
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


def parse_alpha_values(alpha_text: str) -> list[float]:
    parts = [part.strip() for part in alpha_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one alpha value must be provided.")

    alphas = [float(part) for part in parts]
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("All alpha values must be greater than 0.")
    return alphas


def build_feature_target_frames(data: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_name not in data.columns:
        raise ValueError(f"Target column '{target_name}' is not in dataset.")

    feature_columns = [column for column in data.columns if column != target_name]
    features = data[feature_columns]
    target = data[target_name]
    return features, target


def plot_sweep_results(results: pd.DataFrame, save_path: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(results["alpha"], results["rmse"], marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_xscale("log")
    axes[0].set_title("RMSE vs Alpha (Ridge)")
    axes[0].set_xlabel("Alpha (log scale)")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)

    axes[1].plot(results["alpha"], results["coef_l2_norm"], marker="o", linewidth=2, color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_title("Coefficient L2 Norm vs Alpha")
    axes[1].set_xlabel("Alpha (log scale)")
    axes[1].set_ylabel("L2 norm of coefficients")
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        f"Ridge Alpha Sweep (best R^2 = {results['r2'].max():.3f})",
        fontsize=12,
    )
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)


def main() -> None:
    args = parse_args()
    alphas = parse_alpha_values(args.alphas)

    housing_data = load_housing_prices_data(csv_path=args.csv_path)
    features, target = build_feature_target_frames(housing_data, target_name=args.target)

    x_train, x_test, y_train, y_test = split_regression_data(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Scale once, then reuse for every alpha so comparisons stay fair.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)

        rmse = float(root_mean_squared_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        coef_l2_norm = float(np.linalg.norm(model.coef_))

        rows.append(
            {
                "alpha": alpha,
                "rmse": rmse,
                "r2": r2,
                "coef_l2_norm": coef_l2_norm,
            }
        )

    results = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

    print("Housing Ridge Alpha Sweep")
    print("-" * 50)
    print(f"{'Alpha':>10}{'RMSE':>14}{'R^2':>12}{'Coef L2 Norm':>18}")
    print("-" * 50)
    for _, row in results.iterrows():
        print(f"{row['alpha']:>10.4g}{row['rmse']:>14.2f}{row['r2']:>12.3f}{row['coef_l2_norm']:>18.2f}")
    print("-" * 50)

    best_row = results.loc[results["r2"].idxmax()]
    print(
        f"Best by R^2: alpha={best_row['alpha']:.4g}, "
        f"RMSE={best_row['rmse']:.2f}, R^2={best_row['r2']:.3f}"
    )

    plot_sweep_results(results=results, save_path=args.output_path)
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
