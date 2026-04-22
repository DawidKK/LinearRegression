import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Logistic Regression with StandardScaler for feature scaling."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for the main train/test split so results are reproducible.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=10,
        help="How many random states to test for the stability check.",
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/logistic_regression/step2_logistic_regression_with_scaling.png",
        help="Where to save the summary figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure in an interactive window after saving.",
    )
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # Keep the same metric set as Step 1 so comparisons are apples-to-apples.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def run_single_experiment(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    use_scaling: bool,
) -> dict[str, object]:
    # We keep the exact same split logic as Step 1 so only one thing changes per experiment.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    if use_scaling:
        # Step 2 change: fit scaler on training data only, then apply to both train and test.
        # This avoids leaking information from test data into training.
        scaler = StandardScaler()
        x_train_used = scaler.fit_transform(x_train)
        x_test_used = scaler.transform(x_test)
    else:
        # Baseline branch is used only for direct Step 1 vs Step 2 comparison.
        scaler = None
        x_train_used = x_train
        x_test_used = x_test

    # We keep default LogisticRegression settings to isolate the effect of scaling.
    model = LogisticRegression()
    model.fit(x_train_used, y_train)

    # We collect class labels and probabilities because both are required in this learning series.
    y_pred = model.predict(x_test_used)
    y_proba = model.predict_proba(x_test_used)[:, 1]
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    return {
        "model": model,
        "scaler": scaler,
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def summarize_stability(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    stability_runs: int,
    use_scaling: bool,
) -> pd.DataFrame:
    # Stability check repeats training with different random states and records metric movement.
    rows: list[dict[str, float]] = []
    for random_state in range(stability_runs):
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            use_scaling=use_scaling,
        )
        row = {"random_state": float(random_state)}
        row.update(result["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)


def plot_step2_summary(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    output_path: str,
) -> None:
    # The same 3-panel layout keeps visual comparison simple across steps.
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: confusion matrix shows where class mistakes happen.
    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    # Panel 2: ROC curve shows ranking quality across all thresholds.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#1f77b4", linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(alpha=0.25)

    # Panel 3: coefficient magnitudes show how important each standardized feature is.
    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=True)
    axes[2].barh(coefficient_frame["feature"], coefficient_frame["abs_coefficient"], color="#2ca02c")
    axes[2].set_title("Coefficient Magnitude |coef|")
    axes[2].set_xlabel("Absolute coefficient value")
    axes[2].set_ylabel("Feature")
    axes[2].grid(axis="x", alpha=0.25)

    figure.tight_layout()

    # Save artifact for later side-by-side review with Step 1.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def print_metric_comparison(step1_metrics: dict[str, float], step2_metrics: dict[str, float]) -> None:
    # This explicit comparison reinforces what changed and whether it improved.
    print("\nStep 1 vs Step 2 metric comparison (same split):")
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        step1_value = step1_metrics[metric_name]
        step2_value = step2_metrics[metric_name]
        delta = step2_value - step1_value
        print(f"{metric_name:<10}: step1={step1_value:.4f}, step2={step2_value:.4f}, delta={delta:+.4f}")


def main() -> None:
    args = parse_args()

    # Project rule: use only the breast cancer dataset for this logistic regression learning track.
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    # Step 1-equivalent run on the same split is used as a direct comparison anchor.
    baseline_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        use_scaling=False,
    )

    # Step 2 run changes only one thing: we scale features before training.
    scaled_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        use_scaling=True,
    )

    # We evaluate stability for Step 2 and also compare against Step 1 stability.
    baseline_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        use_scaling=False,
    )
    scaled_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        use_scaling=True,
    )

    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    baseline_stability_stats = baseline_stability[metric_columns].agg(["mean", "std"]).T
    scaled_stability_stats = scaled_stability[metric_columns].agg(["mean", "std"]).T

    print("Step 2 - Logistic Regression with Feature Scaling (StandardScaler)")
    print("-" * 66)
    print(f"Dataset: {dataset.frame.shape[0]} rows, {features.shape[1]} features")
    print(f"Train/test split: {(1 - args.test_size):.0%}/{args.test_size:.0%}")
    print(f"Main random_state: {args.random_state}")

    print("\nMain Step 2 metrics (single split):")
    for metric_name, metric_value in scaled_result["metrics"].items():
        print(f"{metric_name:<10}: {metric_value:.4f}")

    print_metric_comparison(step1_metrics=baseline_result["metrics"], step2_metrics=scaled_result["metrics"])

    print("\nStability check comparison (random_state 0 to " f"{args.stability_runs - 1}):")
    for metric_name in metric_columns:
        base_mean = float(baseline_stability_stats.loc[metric_name, "mean"])
        base_std = float(baseline_stability_stats.loc[metric_name, "std"])
        scaled_mean = float(scaled_stability_stats.loc[metric_name, "mean"])
        scaled_std = float(scaled_stability_stats.loc[metric_name, "std"])
        std_delta = scaled_std - base_std
        print(
            f"{metric_name:<10}: "
            f"step1 mean={base_mean:.4f}, std={base_std:.4f} | "
            f"step2 mean={scaled_mean:.4f}, std={scaled_std:.4f}, std delta={std_delta:+.4f}"
        )

    # Interpretation keeps the learning focus on the "why" behind scaling behavior.
    print("\nInterpretation:")
    print("Scaling changes coefficient units to standard deviations, so magnitudes become more comparable.")
    print("Performance may change a little because optimization is usually easier on similarly scaled features.")
    print("Step 3 will keep scaling and then tune regularization strength (C).")

    print("\nTop 10 features by |coefficient| in Step 2 model (scaled features):")
    scaled_coefficients = scaled_result["model"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {"feature": features.columns, "coefficient": scaled_coefficients, "abs_coef": np.abs(scaled_coefficients)}
    ).sort_values("abs_coef", ascending=False)
    for _, row in coefficient_frame.head(10).iterrows():
        print(f"{row['feature']:<30} coef={row['coefficient']:+.4f}")

    plot_step2_summary(
        y_test=scaled_result["y_test"],
        y_pred=scaled_result["y_pred"],
        y_proba=scaled_result["y_proba"],
        feature_names=list(features.columns),
        coefficients=scaled_coefficients,
        output_path=args.output_path,
    )
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
