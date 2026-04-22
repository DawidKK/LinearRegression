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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Raw Logistic Regression baseline (no scaling, default model settings)."
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
        default="reports/figures/logistic_regression/step1_raw_logistic_regression_baseline.png",
        help="Where to save the summary figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure in an interactive window after saving.",
    )
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # Keep all required classification metrics in one place so every experiment uses the same score set.
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
) -> dict[str, object]:
    # Step 1 starts with a plain train/test split because this is the minimum setup for honest evaluation.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # Baseline rule: use default LogisticRegression settings and no feature scaling.
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # We collect both hard classes and probabilities because later steps will tune thresholds.
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    # Metrics are computed from the test split only to reflect unseen-data behavior.
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    return {
        "model": model,
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
) -> pd.DataFrame:
    # Stability check: repeat training with different random states to measure how much metrics move.
    rows: list[dict[str, float]] = []
    for random_state in range(stability_runs):
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
        )
        row = {"random_state": float(random_state)}
        row.update(result["metrics"])
        rows.append(row)

    stability_frame = pd.DataFrame(rows)
    return stability_frame


def plot_step1_summary(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    output_path: str,
) -> None:
    # One figure with three panels makes Step 1 easy to scan and compare against future steps.
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: confusion matrix shows which class is being confused with which.
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

    # Panel 2: ROC curve shows how class separation changes across all possible thresholds.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#1f77b4", linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(alpha=0.25)

    # Panel 3: absolute coefficient size shows which features influence the decision most.
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

    # Save the figure so Step 1 artifacts are preserved for later side-by-side comparison.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main() -> None:
    args = parse_args()

    # The project rule says to use only this dataset for logistic regression practice.
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    # Main baseline run: this is the reference point for every later step.
    baseline_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    baseline_metrics = baseline_result["metrics"]

    # Stability run: we repeat the same setup with different random states to inspect variance.
    stability_frame = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
    )
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    stability_stats = stability_frame[metric_columns].agg(["mean", "std"]).T

    print("Step 1 - Raw Logistic Regression (Baseline)")
    print("-" * 48)
    print(f"Dataset: {dataset.frame.shape[0]} rows, {features.shape[1]} features")
    print(f"Train/test split: {(1 - args.test_size):.0%}/{args.test_size:.0%}")
    print(f"Main random_state: {args.random_state}")

    print("\nMain baseline metrics (single split):")
    for metric_name, metric_value in baseline_metrics.items():
        print(f"{metric_name:<10}: {metric_value:.4f}")

    print("\nStability check across random states:")
    print(f"Runs: {args.stability_runs} (random_state 0 to {args.stability_runs - 1})")
    for metric_name in metric_columns:
        mean_value = float(stability_stats.loc[metric_name, "mean"])
        std_value = float(stability_stats.loc[metric_name, "std"])
        print(f"{metric_name:<10}: mean={mean_value:.4f}, std={std_value:.4f}")

    # Simple interpretation text helps learning: low std means stable behavior across splits.
    max_std_metric = stability_stats["std"].idxmax()
    max_std_value = float(stability_stats.loc[max_std_metric, "std"])
    print("\nInterpretation:")
    if max_std_value < 0.02:
        print("Model looks stable because metric variation across random states is low.")
    else:
        print("Model shows noticeable variation, so later steps should check if stability improves.")
    print("This is our baseline reference for Step 2 (adding feature scaling).")

    print("\nTop 10 features by |coefficient| in baseline model:")
    coefficients = baseline_result["model"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {"feature": features.columns, "coefficient": coefficients, "abs_coef": np.abs(coefficients)}
    ).sort_values("abs_coef", ascending=False)
    for _, row in coefficient_frame.head(10).iterrows():
        print(f"{row['feature']:<30} coef={row['coefficient']:+.4f}")

    plot_step1_summary(
        y_test=baseline_result["y_test"],
        y_pred=baseline_result["y_pred"],
        y_proba=baseline_result["y_proba"],
        feature_names=list(features.columns),
        coefficients=coefficients,
        output_path=args.output_path,
    )
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
