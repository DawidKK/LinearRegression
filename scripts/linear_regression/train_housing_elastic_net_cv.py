import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from ml_training_practice.data import load_housing_prices_data
from ml_training_practice.preprocessing import split_regression_data


# This script demonstrates ElasticNetCV on a small housing dataset.
# Why Elastic Net:
# - L1 part (like Lasso) can set weak feature weights to exactly 0 -> sparse model.
# - L2 part (like Ridge) shrinks all coefficients smoothly -> better stability.
# Why ElasticNetCV:
# - It runs cross-validation internally and picks the best alpha + l1_ratio for us.
# - This lets us study model selection without writing manual nested loops.


def parse_args() -> argparse.Namespace:
    # Command-line arguments make the experiment configurable without editing code.
    # This is useful when you want to quickly rerun the same script with different
    # regularization ranges or CV settings.
    parser = argparse.ArgumentParser(
        description="Train ElasticNetCV on housing data and visualize CV behavior."
    )
    parser.add_argument(
        "--target",
        default="price_usd",
        help="Target column name.",
    )
    parser.add_argument(
        "--l1-ratios",
        default="0.1,0.3,0.5,0.7,0.9,0.95,0.99,1.0",
        help=(
            "Comma-separated l1_ratio values to test. "
            "0 = pure Ridge, 1 = pure Lasso."
        ),
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=1e-3,
        help="Smallest alpha in the tested grid.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=1e2,
        help="Largest alpha in the tested grid.",
    )
    parser.add_argument(
        "--alpha-count",
        type=int,
        default=40,
        help="How many alpha values to test between alpha-min and alpha-max (log-spaced).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of folds for cross-validation.",
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
        help="Random seed for train/test split and ElasticNetCV.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Maximum optimization iterations for ElasticNet solver.",
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/housing_elastic_net_cv.png",
        help="Where to save the summary figure (.png).",
    )
    parser.add_argument(
        "--csv-path",
        default="data/raw/housing_prices_sample.csv",
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated figure in an interactive window.",
    )
    return parser.parse_args()


def parse_l1_ratios(ratios_text: str) -> list[float]:
    # Convert user input like "0.1,0.5,0.9" into numeric values.
    # Validation avoids silent bugs (for example invalid values like -0.5 or 2.0).
    parts = [part.strip() for part in ratios_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one l1_ratio value must be provided.")

    values = [float(part) for part in parts]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("All l1_ratio values must be in [0, 1].")
    return values


def build_alpha_grid(alpha_min: float, alpha_max: float, alpha_count: int) -> np.ndarray:
    # Elastic Net regularization strength is controlled by alpha.
    # We use log spacing because useful alpha values usually span several orders
    # of magnitude (e.g. 0.001 to 100), not a narrow linear range.
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("alpha-min and alpha-max must be greater than 0.")
    if alpha_min >= alpha_max:
        raise ValueError("alpha-min must be smaller than alpha-max.")
    if alpha_count < 2:
        raise ValueError("alpha-count must be at least 2.")

    return np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_count)


def build_feature_target_frames(data: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    # Separate predictors (X) from target (y) explicitly.
    # This keeps data flow readable for beginners and makes shape mistakes easier to catch.
    if target_name not in data.columns:
        raise ValueError(f"Target column '{target_name}' is not in dataset.")

    feature_columns = [column for column in data.columns if column != target_name]
    features = data[feature_columns]
    target = data[target_name]
    return features, target


def build_cv_summary(model: ElasticNetCV) -> pd.DataFrame:
    # This helper converts internal CV data from scikit-learn into a clean table.
    # The table shows, for each l1_ratio, which alpha had the best average CV RMSE.
    # That makes model-selection behavior easy to inspect in the console.
    # mse_path_ shape:
    # - multiple l1_ratio values: (n_l1_ratios, n_alphas, n_folds)
    # - single l1_ratio value: (n_alphas, n_folds)
    mse_path = model.mse_path_
    if mse_path.ndim == 2:
        mse_path = np.expand_dims(mse_path, axis=0)

    # model.l1_ratio can be scalar or array depending on input.
    # np.atleast_1d gives us one consistent format for looping.
    l1_values = np.atleast_1d(model.l1_ratio)
    alpha_values = model.alphas_

    rows: list[dict[str, float]] = []
    for idx, l1_value in enumerate(l1_values):
        # Average across folds: one mean CV error for each alpha candidate.
        mean_mse_by_alpha = mse_path[idx].mean(axis=1)
        best_idx = int(np.argmin(mean_mse_by_alpha))

        rows.append(
            {
                "l1_ratio": float(l1_value),
                "best_alpha_for_l1_ratio": float(alpha_values[best_idx]),
                "best_cv_rmse_for_l1_ratio": float(np.sqrt(mean_mse_by_alpha[best_idx])),
            }
        )

    return pd.DataFrame(rows).sort_values("l1_ratio").reset_index(drop=True)


def plot_results(model: ElasticNetCV, feature_names: pd.Index, save_path: str) -> None:
    # Create one figure with two views:
    # 1) CV error curves to understand hyperparameter selection.
    # 2) Final coefficients to understand feature impact and sparsity.
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    mse_path = model.mse_path_
    if mse_path.ndim == 2:
        mse_path = np.expand_dims(mse_path, axis=0)

    l1_values = np.atleast_1d(model.l1_ratio)
    alpha_values = model.alphas_

    # Left plot:
    # For each l1_ratio candidate, draw RMSE as alpha changes.
    # This helps explain which region of regularization works best.
    for idx, l1_value in enumerate(l1_values):
        mean_rmse_by_alpha = np.sqrt(mse_path[idx].mean(axis=1))
        axes[0].plot(
            alpha_values,
            mean_rmse_by_alpha,
            linewidth=2,
            label=f"l1_ratio={l1_value:.2f}",
        )

    # Vertical line marks alpha selected by ElasticNetCV.
    axes[0].axvline(model.alpha_, color="black", linestyle="--", linewidth=1.5, label="selected alpha")
    axes[0].set_xscale("log")
    axes[0].set_title("Cross-Validated RMSE vs Alpha")
    axes[0].set_xlabel("Alpha (log scale)")
    axes[0].set_ylabel("CV RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    # Right plot:
    # Sort coefficients by absolute size so strongest effects appear first.
    # Blue = positive relation, red = negative relation.
    coefficient_series = pd.Series(model.coef_, index=feature_names)
    coefficient_series = coefficient_series.reindex(coefficient_series.abs().sort_values(ascending=False).index)

    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in coefficient_series]
    axes[1].bar(coefficient_series.index, coefficient_series.values, color=colors)
    axes[1].set_title("Elastic Net Coefficients (sorted)")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Coefficient value")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)

    # Count non-zero coefficients to highlight how sparse the final model is.
    non_zero = int(np.sum(~np.isclose(model.coef_, 0.0)))
    total = len(model.coef_)
    figure.suptitle(
        (
            "ElasticNetCV on Housing Data "
            f"(selected l1_ratio={model.l1_ratio_:.2f}, alpha={model.alpha_:.4g}, "
            f"non-zero coefficients={non_zero}/{total})"
        ),
        fontsize=12,
    )

    figure.tight_layout()

    # Ensure output directory exists before writing file.
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)


def main() -> None:
    # Parse and validate experiment setup from CLI.
    args = parse_args()
    l1_ratios = parse_l1_ratios(args.l1_ratios)
    alpha_grid = build_alpha_grid(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_count=args.alpha_count,
    )

    # 1) Load data and split into features (X) and target (y).
    # Keeping this explicit helps learners see how tabular regression input is formed.
    housing_data = load_housing_prices_data(csv_path=args.csv_path)
    features, target = build_feature_target_frames(housing_data, target_name=args.target)

    # 2) Hold out test data BEFORE training.
    # Test set simulates unseen data and gives a more honest performance check.
    x_train, x_test, y_train, y_test = split_regression_data(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 3) Scale features because regularization is sensitive to feature units.
    # Important rule: fit scaler on training data only, then transform test data.
    # This avoids information leakage from test into training.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 4) Train ElasticNetCV.
    # It evaluates many (l1_ratio, alpha) combinations with K-fold CV and picks
    # the combination that minimizes validation error.
    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alpha_grid,
        cv=args.cv,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    model.fit(x_train_scaled, y_train)

    # 5) Evaluate on held-out test split.
    # RMSE: error in target units (price), easier to interpret magnitude.
    # R^2: proportion of variance explained (closer to 1 is better).
    predictions = model.predict(x_test_scaled)
    rmse = float(root_mean_squared_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    # Inspect coefficient behavior to understand regularization effects.
    coefficient_series = pd.Series(model.coef_, index=features.columns)
    zero_coefficients = int(np.sum(np.isclose(model.coef_, 0.0)))
    coef_l1_norm = float(np.linalg.norm(model.coef_, ord=1))
    coef_l2_norm = float(np.linalg.norm(model.coef_, ord=2))

    # Build readable CV summary for console output.
    cv_summary = build_cv_summary(model)

    print("Housing ElasticNetCV")
    print("-" * 72)
    print(f"Selected l1_ratio: {model.l1_ratio_:.4f}")
    print(f"Selected alpha:    {model.alpha_:.6g}")
    print(f"Test RMSE:         {rmse:.2f}")
    print(f"Test R^2:          {r2:.3f}")
    print(f"Zero coefficients: {zero_coefficients}/{len(model.coef_)}")
    print(f"Coefficient L1 norm: {coef_l1_norm:.2f}")
    print(f"Coefficient L2 norm: {coef_l2_norm:.2f}")
    print("-" * 72)

    print("\nBest CV result inside each l1_ratio bucket")
    print("-" * 72)
    print(f"{'l1_ratio':>10}{'best_alpha':>16}{'best_cv_rmse':>16}")
    print("-" * 72)
    for _, row in cv_summary.iterrows():
        print(
            f"{row['l1_ratio']:>10.2f}"
            f"{row['best_alpha_for_l1_ratio']:>16.6g}"
            f"{row['best_cv_rmse_for_l1_ratio']:>16.2f}"
        )
    print("-" * 72)

    # Sort coefficients by absolute value so the most influential features are first.
    print("\nFinal model coefficients")
    print(coefficient_series.sort_values(key=np.abs, ascending=False))

    # Save visualization report (figure file) for later comparison with other models.
    plot_results(model=model, feature_names=features.columns, save_path=args.output_path)
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
