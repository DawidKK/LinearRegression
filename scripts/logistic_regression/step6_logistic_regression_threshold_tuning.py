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
        description="Step 6: Threshold tuning using the Step 5 Elastic Net model."
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
        "--c-values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="C values for the Elastic Net model (same search space as Step 5).",
    )
    parser.add_argument(
        "--threshold-values",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Candidate thresholds to evaluate.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["precision", "recall", "f1_score", "accuracy"],
        default="f1_score",
        help="Metric used to choose the best threshold.",
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/logistic_regression/step6_logistic_regression_threshold_tuning.png",
        help="Where to save the summary figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure in an interactive window after saving.",
    )
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # We keep the same metric set used in earlier steps for direct comparability.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def predict_with_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    # Threshold converts probabilities into class labels.
    # Lower threshold usually raises recall; higher threshold usually raises precision.
    return (y_proba >= threshold).astype(int)


def build_elastic_net_model(c_value: float) -> LogisticRegression:
    # Step 6 keeps exactly the Step 5 model setup and changes only the decision threshold.
    return LogisticRegression(
        C=c_value,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=5000,
    )


def run_single_experiment(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    c_value: float,
    threshold: float,
) -> dict[str, object]:
    # We keep the same split structure to ensure controlled, comparable experiments.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # We keep scaling from Step 2 onward because regularization behavior depends on feature scale.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # We train the Step 5 Elastic Net model (same regularization settings as previous step).
    model = build_elastic_net_model(c_value=c_value)
    model.fit(x_train_scaled, y_train)

    # We use probabilities and then apply a configurable threshold.
    y_proba = model.predict_proba(x_test_scaled)[:, 1]
    y_pred = predict_with_threshold(y_proba=y_proba, threshold=threshold)
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    return {
        "model": model,
        "scaler": scaler,
        "x_test": x_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def run_c_sweep(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    c_values: list[float],
) -> pd.DataFrame:
    # Step 6 keeps Step 5 model selection by C; threshold stays at 0.5 during C search.
    rows: list[dict[str, float]] = []
    for c_value in c_values:
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            c_value=c_value,
            threshold=0.5,
        )
        coefficients = result["model"].coef_[0]
        rows.append(
            {
                "C": float(c_value),
                "accuracy": result["metrics"]["accuracy"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1_score": result["metrics"]["f1_score"],
                "roc_auc": result["metrics"]["roc_auc"],
                "coef_l2_norm": float(np.linalg.norm(coefficients, ord=2)),
                "zero_count": float(coefficients.size - np.count_nonzero(coefficients)),
            }
        )
    return pd.DataFrame(rows).sort_values("C")


def evaluate_thresholds(y_true: pd.Series, y_proba: np.ndarray, threshold_values: list[float]) -> pd.DataFrame:
    # We evaluate many thresholds on the same model output to isolate threshold effect only.
    rows: list[dict[str, float]] = []
    for threshold in threshold_values:
        y_pred = predict_with_threshold(y_proba=y_proba, threshold=threshold)
        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows).sort_values("threshold")


def summarize_stability(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    stability_runs: int,
    c_value: float,
    threshold: float,
) -> pd.DataFrame:
    # Stability check: repeat training across random states with one fixed threshold.
    rows: list[dict[str, float]] = []
    for random_state in range(stability_runs):
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            c_value=c_value,
            threshold=threshold,
        )
        row = {"random_state": float(random_state)}
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_step6_summary(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    feature_names: list[str],
    coefficients: np.ndarray,
    output_path: str,
) -> None:
    # We keep the same three required panels to make side-by-side step comparison easy.
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: confusion matrix reflects predictions after threshold tuning.
    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    # Panel 2: ROC curve still comes from probabilities and is threshold-independent.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#1f77b4", linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(alpha=0.25)

    # Panel 3: coefficient magnitudes still explain feature influence in the underlying model.
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

    # Save figure so Step 6 artifacts can be compared with prior steps.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main() -> None:
    args = parse_args()

    # We use only the breast cancer dataset, following project rules.
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    # Step 5 model selection: choose best C at threshold 0.5.
    c_sweep = run_c_sweep(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_values=args.c_values,
    )
    best_c_row = c_sweep.sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False).iloc[0]
    best_c = float(best_c_row["C"])

    # We run one baseline pass with threshold 0.5 for direct Step 5 comparison.
    step5_like_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_value=best_c,
        threshold=0.5,
    )

    # Step 6 core action: evaluate multiple thresholds on the same probability outputs.
    threshold_candidates = sorted(set(args.threshold_values + [0.5]))
    threshold_frame = evaluate_thresholds(
        y_true=step5_like_result["y_test"],
        y_proba=step5_like_result["y_proba"],
        threshold_values=threshold_candidates,
    )
    best_threshold_row = threshold_frame.sort_values(
        [args.selection_metric, "recall", "precision", "accuracy"], ascending=False
    ).iloc[0]
    best_threshold = float(best_threshold_row["threshold"])

    # We create final Step 6 predictions with the selected threshold.
    step6_y_pred = predict_with_threshold(y_proba=step5_like_result["y_proba"], threshold=best_threshold)
    step6_metrics = compute_metrics(
        y_true=step5_like_result["y_test"],
        y_pred=step6_y_pred,
        y_proba=step5_like_result["y_proba"],
    )

    # Stability comparison checks default threshold vs tuned threshold across random states.
    default_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        c_value=best_c,
        threshold=0.5,
    )
    tuned_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        c_value=best_c,
        threshold=best_threshold,
    )

    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    default_stability_stats = default_stability[metric_columns].agg(["mean", "std"]).T
    tuned_stability_stats = tuned_stability[metric_columns].agg(["mean", "std"]).T

    print("Step 6 - Threshold Tuning (Elastic Net model from Step 5)")
    print("-" * 58)
    print(f"Dataset: {dataset.frame.shape[0]} rows, {features.shape[1]} features")
    print(f"Train/test split: {(1 - args.test_size):.0%}/{args.test_size:.0%}")
    print(f"Main random_state: {args.random_state}")
    print(f"Selected C (from Step 5 setup): {best_c:.2f}")

    print("\nThreshold sweep results:")
    for _, row in threshold_frame.iterrows():
        print(
            f"thr={row['threshold']:.2f} "
            f"acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} "
            f"rec={row['recall']:.4f} "
            f"f1={row['f1_score']:.4f} "
            f"roc_auc={row['roc_auc']:.4f}"
        )

    print(f"\nSelected threshold for Step 6 ({args.selection_metric}): {best_threshold:.2f}")

    print("\nStep 5 vs Step 6 metric comparison (same model, same split):")
    for metric_name in metric_columns:
        step5_value = step5_like_result["metrics"][metric_name]
        step6_value = step6_metrics[metric_name]
        print(f"{metric_name:<10}: step5={step5_value:.4f}, step6={step6_value:.4f}, delta={step6_value-step5_value:+.4f}")

    print("\nStability check comparison (random_state 0 to " f"{args.stability_runs - 1}):")
    for metric_name in metric_columns:
        default_mean = float(default_stability_stats.loc[metric_name, "mean"])
        default_std = float(default_stability_stats.loc[metric_name, "std"])
        tuned_mean = float(tuned_stability_stats.loc[metric_name, "mean"])
        tuned_std = float(tuned_stability_stats.loc[metric_name, "std"])
        print(
            f"{metric_name:<10}: "
            f"default mean={default_mean:.4f}, std={default_std:.4f} | "
            f"tuned mean={tuned_mean:.4f}, std={tuned_std:.4f}, std delta={tuned_std - default_std:+.4f}"
        )

    # Interpretation explains why changing threshold affects precision and recall.
    print("\nInterpretation:")
    print("Threshold changes only the final class decision rule, not the model probabilities.")
    print("Lower threshold usually catches more positives (higher recall) but can add false positives.")
    print("Higher threshold usually reduces false positives (higher precision) but can miss positives.")

    print("\nTop 10 features by |coefficient| in Step 6 model:")
    coefficients = step5_like_result["model"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {"feature": features.columns, "coefficient": coefficients, "abs_coef": np.abs(coefficients)}
    ).sort_values("abs_coef", ascending=False)
    for _, row in coefficient_frame.head(10).iterrows():
        print(f"{row['feature']:<30} coef={row['coefficient']:+.4f}")

    plot_step6_summary(
        y_test=step5_like_result["y_test"],
        y_pred=step6_y_pred,
        y_proba=step5_like_result["y_proba"],
        threshold=best_threshold,
        feature_names=list(features.columns),
        coefficients=coefficients,
        output_path=args.output_path,
    )
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
