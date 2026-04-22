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
        description="Step 3: L2 regularization tuning by changing C while keeping feature scaling."
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
        help="C values to test. Smaller C means stronger regularization.",
    )
    parser.add_argument(
        "--output-path",
        default="reports/figures/logistic_regression/step3_logistic_regression_l2_c_tuning.png",
        help="Where to save the summary figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure in an interactive window after saving.",
    )
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # We keep the same metric set used in previous steps so comparisons stay fair.
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
    c_value: float,
) -> dict[str, object]:
    # We use the same train/test split structure as Step 2 for one-variable-at-a-time learning.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # Step 3 keeps scaling from Step 2 because regularization is sensitive to feature units.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Step 3 change: tune L2 strength with C. Small C means stronger shrinkage.
    # We keep default penalty behavior (L2) and change only C for clean comparison.
    model = LogisticRegression(C=c_value)
    model.fit(x_train_scaled, y_train)

    # Predict class labels and probabilities so all required evaluation pieces are available.
    y_pred = model.predict(x_test_scaled)
    y_proba = model.predict_proba(x_test_scaled)[:, 1]
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


def run_c_sweep(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    c_values: list[float],
) -> pd.DataFrame:
    # We train one model per C so we can see underfitting-to-overfitting behavior.
    rows: list[dict[str, float]] = []
    for c_value in c_values:
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            c_value=c_value,
        )
        coefficients = result["model"].coef_[0]
        row = {
            "C": float(c_value),
            "accuracy": result["metrics"]["accuracy"],
            "precision": result["metrics"]["precision"],
            "recall": result["metrics"]["recall"],
            "f1_score": result["metrics"]["f1_score"],
            "roc_auc": result["metrics"]["roc_auc"],
            "coef_l2_norm": float(np.linalg.norm(coefficients, ord=2)),
            "coef_abs_mean": float(np.abs(coefficients).mean()),
        }
        rows.append(row)

    sweep_frame = pd.DataFrame(rows).sort_values("C")
    return sweep_frame


def summarize_stability(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    stability_runs: int,
    c_value: float,
) -> pd.DataFrame:
    # Stability check repeats the same setup with different random states.
    rows: list[dict[str, float]] = []
    for random_state in range(stability_runs):
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            c_value=c_value,
        )
        row = {"random_state": float(random_state)}
        row.update(result["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)


def plot_step3_summary(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    output_path: str,
) -> None:
    # We keep the same 3-panel plot used in earlier steps for easy visual comparison.
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: confusion matrix highlights where the classifier is making mistakes.
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

    # Panel 3: coefficient magnitudes show how much L2 regularization is shrinking weights.
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

    # Save artifact so Step 3 can be reviewed side-by-side with earlier steps.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main() -> None:
    args = parse_args()

    # Project rule: use only this dataset for logistic regression experiments.
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    # Step 2 baseline uses C=1.0 with scaling, so this is our direct comparison anchor.
    step2_like_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_value=1.0,
    )

    # Step 3 core action: sweep C values and choose the best value on the same split.
    sweep_frame = run_c_sweep(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_values=args.c_values,
    )
    best_row = sweep_frame.sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False).iloc[0]
    best_c = float(best_row["C"])

    # We re-run once with best C to collect predictions and coefficients for reporting and plotting.
    step3_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_value=best_c,
    )

    # Stability comparison checks if tuned regularization is more consistent across random states.
    step2_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        c_value=1.0,
    )
    step3_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        c_value=best_c,
    )

    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    step2_stability_stats = step2_stability[metric_columns].agg(["mean", "std"]).T
    step3_stability_stats = step3_stability[metric_columns].agg(["mean", "std"]).T

    print("Step 3 - Logistic Regression with L2 Regularization (C tuning)")
    print("-" * 63)
    print(f"Dataset: {dataset.frame.shape[0]} rows, {features.shape[1]} features")
    print(f"Train/test split: {(1 - args.test_size):.0%}/{args.test_size:.0%}")
    print(f"Main random_state: {args.random_state}")

    print("\nC sweep results (same split):")
    for _, row in sweep_frame.iterrows():
        print(
            f"C={row['C']:<7.2f} "
            f"acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} "
            f"rec={row['recall']:.4f} "
            f"f1={row['f1_score']:.4f} "
            f"roc_auc={row['roc_auc']:.4f} "
            f"coef_l2={row['coef_l2_norm']:.4f}"
        )

    print(f"\nSelected C for Step 3: {best_c:.2f}")

    print("\nStep 2 vs Step 3 metric comparison (same split):")
    for metric_name in metric_columns:
        step2_value = step2_like_result["metrics"][metric_name]
        step3_value = step3_result["metrics"][metric_name]
        delta = step3_value - step2_value
        print(f"{metric_name:<10}: step2={step2_value:.4f}, step3={step3_value:.4f}, delta={delta:+.4f}")

    step2_coef = step2_like_result["model"].coef_[0]
    step3_coef = step3_result["model"].coef_[0]
    step2_coef_l2 = float(np.linalg.norm(step2_coef, ord=2))
    step3_coef_l2 = float(np.linalg.norm(step3_coef, ord=2))
    print("\nCoefficient shrinkage check (L2 norm):")
    print(f"step2 C=1.00 coef_l2_norm : {step2_coef_l2:.4f}")
    print(f"step3 C={best_c:.2f} coef_l2_norm: {step3_coef_l2:.4f}")
    print(f"delta (step3 - step2)      : {step3_coef_l2 - step2_coef_l2:+.4f}")

    print("\nStability check comparison (random_state 0 to " f"{args.stability_runs - 1}):")
    for metric_name in metric_columns:
        step2_mean = float(step2_stability_stats.loc[metric_name, "mean"])
        step2_std = float(step2_stability_stats.loc[metric_name, "std"])
        step3_mean = float(step3_stability_stats.loc[metric_name, "mean"])
        step3_std = float(step3_stability_stats.loc[metric_name, "std"])
        print(
            f"{metric_name:<10}: "
            f"step2 mean={step2_mean:.4f}, std={step2_std:.4f} | "
            f"step3 mean={step3_mean:.4f}, std={step3_std:.4f}, std delta={step3_std - step2_std:+.4f}"
        )

    # Interpretation focuses on why C changes both coefficients and model behavior.
    print("\nInterpretation:")
    print("Smaller C means stronger regularization, which shrinks coefficients toward zero.")
    print("Too much shrinkage can underfit. Too little shrinkage can overfit.")
    print("Our tuned C balances these effects for this split and dataset.")

    print("\nTop 10 features by |coefficient| in Step 3 model:")
    coefficient_frame = pd.DataFrame(
        {"feature": features.columns, "coefficient": step3_coef, "abs_coef": np.abs(step3_coef)}
    ).sort_values("abs_coef", ascending=False)
    for _, row in coefficient_frame.head(10).iterrows():
        print(f"{row['feature']:<30} coef={row['coefficient']:+.4f}")

    plot_step3_summary(
        y_test=step3_result["y_test"],
        y_pred=step3_result["y_pred"],
        y_proba=step3_result["y_proba"],
        feature_names=list(features.columns),
        coefficients=step3_coef,
        output_path=args.output_path,
    )
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
