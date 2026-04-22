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
        description="Step 4: L1 regularization (Lasso) with solver='liblinear' and C tuning."
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
        default="reports/figures/logistic_regression/step4_logistic_regression_l1_lasso.png",
        help="Where to save the summary figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure in an interactive window after saving.",
    )
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # We keep the same metrics as earlier steps so comparisons remain fair.
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
    penalty_name: str,
    solver_name: str,
) -> dict[str, object]:
    # We keep the same split strategy as Step 3 so only regularization type changes.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # We keep StandardScaler from Step 2 onward because regularization depends on feature scale.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Step 4 main change: use L1 penalty with liblinear solver.
    model = LogisticRegression(C=c_value, penalty=penalty_name, solver=solver_name)
    model.fit(x_train_scaled, y_train)

    # We gather class labels and probabilities for full evaluation and plots.
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
    penalty_name: str,
    solver_name: str,
) -> pd.DataFrame:
    # We train one model per C to see the tradeoff between shrinkage and predictive quality.
    rows: list[dict[str, float]] = []
    for c_value in c_values:
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            c_value=c_value,
            penalty_name=penalty_name,
            solver_name=solver_name,
        )
        coefficients = result["model"].coef_[0]
        non_zero_count = int(np.count_nonzero(coefficients))
        zero_count = int(coefficients.size - non_zero_count)
        row = {
            "C": float(c_value),
            "accuracy": result["metrics"]["accuracy"],
            "precision": result["metrics"]["precision"],
            "recall": result["metrics"]["recall"],
            "f1_score": result["metrics"]["f1_score"],
            "roc_auc": result["metrics"]["roc_auc"],
            "coef_l2_norm": float(np.linalg.norm(coefficients, ord=2)),
            "non_zero_count": float(non_zero_count),
            "zero_count": float(zero_count),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("C")


def summarize_stability(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    stability_runs: int,
    c_value: float,
    penalty_name: str,
    solver_name: str,
) -> pd.DataFrame:
    # Stability check repeats the same setup across random states.
    rows: list[dict[str, float]] = []
    for random_state in range(stability_runs):
        result = run_single_experiment(
            features=features,
            target=target,
            test_size=test_size,
            random_state=random_state,
            c_value=c_value,
            penalty_name=penalty_name,
            solver_name=solver_name,
        )
        row = {"random_state": float(random_state)}
        row.update(result["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)


def plot_step4_summary(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    output_path: str,
) -> None:
    # The same 3-panel view makes visual comparison easy from Step 1 to Step 4.
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: confusion matrix shows true vs predicted classes.
    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    # Panel 2: ROC curve summarizes ranking quality across thresholds.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    axes[1].plot(false_positive_rate, true_positive_rate, color="#1f77b4", linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(alpha=0.25)

    # Panel 3: coefficient magnitudes show feature influence and L1 shrinkage effect.
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

    # Save figure so Step 4 can be reviewed side by side with prior steps.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main() -> None:
    args = parse_args()

    # Project rule: we use only this dataset for all logistic regression steps.
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    # Step 3 reference: L2 with scaling and C tuning.
    step3_sweep = run_c_sweep(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_values=args.c_values,
        penalty_name="l2",
        solver_name="lbfgs",
    )
    step3_best_row = step3_sweep.sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False).iloc[0]
    step3_best_c = float(step3_best_row["C"])
    step3_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_value=step3_best_c,
        penalty_name="l2",
        solver_name="lbfgs",
    )

    # Step 4 core action: L1 (lasso) with required solver and C tuning.
    step4_sweep = run_c_sweep(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_values=args.c_values,
        penalty_name="l1",
        solver_name="liblinear",
    )
    step4_best_row = step4_sweep.sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False).iloc[0]
    step4_best_c = float(step4_best_row["C"])
    step4_result = run_single_experiment(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
        c_value=step4_best_c,
        penalty_name="l1",
        solver_name="liblinear",
    )

    # Stability check compares Step 3 best setup against Step 4 best setup.
    step3_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        c_value=step3_best_c,
        penalty_name="l2",
        solver_name="lbfgs",
    )
    step4_stability = summarize_stability(
        features=features,
        target=target,
        test_size=args.test_size,
        stability_runs=args.stability_runs,
        c_value=step4_best_c,
        penalty_name="l1",
        solver_name="liblinear",
    )

    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    step3_stability_stats = step3_stability[metric_columns].agg(["mean", "std"]).T
    step4_stability_stats = step4_stability[metric_columns].agg(["mean", "std"]).T

    print("Step 4 - Logistic Regression with L1 Regularization (Lasso)")
    print("-" * 58)
    print(f"Dataset: {dataset.frame.shape[0]} rows, {features.shape[1]} features")
    print(f"Train/test split: {(1 - args.test_size):.0%}/{args.test_size:.0%}")
    print(f"Main random_state: {args.random_state}")

    print("\nStep 4 (L1) C sweep results:")
    for _, row in step4_sweep.iterrows():
        print(
            f"C={row['C']:<7.2f} "
            f"acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} "
            f"rec={row['recall']:.4f} "
            f"f1={row['f1_score']:.4f} "
            f"roc_auc={row['roc_auc']:.4f} "
            f"coef_l2={row['coef_l2_norm']:.4f} "
            f"zeros={int(row['zero_count']):>2}/{features.shape[1]}"
        )

    print(f"\nSelected C for Step 3 (L2): {step3_best_c:.2f}")
    print(f"Selected C for Step 4 (L1): {step4_best_c:.2f}")

    print("\nStep 3 vs Step 4 metric comparison (same split):")
    for metric_name in metric_columns:
        step3_value = step3_result["metrics"][metric_name]
        step4_value = step4_result["metrics"][metric_name]
        delta = step4_value - step3_value
        print(f"{metric_name:<10}: step3={step3_value:.4f}, step4={step4_value:.4f}, delta={delta:+.4f}")

    step3_coef = step3_result["model"].coef_[0]
    step4_coef = step4_result["model"].coef_[0]
    step4_non_zero = int(np.count_nonzero(step4_coef))
    step4_zero = int(step4_coef.size - step4_non_zero)
    print("\nSparsity check (Step 4 L1):")
    print(f"non-zero coefficients: {step4_non_zero}/{step4_coef.size}")
    print(f"zero coefficients    : {step4_zero}/{step4_coef.size}")
    print(f"Step 3 coef_l2_norm : {np.linalg.norm(step3_coef, ord=2):.4f}")
    print(f"Step 4 coef_l2_norm : {np.linalg.norm(step4_coef, ord=2):.4f}")

    print("\nStability check comparison (random_state 0 to " f"{args.stability_runs - 1}):")
    for metric_name in metric_columns:
        step3_mean = float(step3_stability_stats.loc[metric_name, "mean"])
        step3_std = float(step3_stability_stats.loc[metric_name, "std"])
        step4_mean = float(step4_stability_stats.loc[metric_name, "mean"])
        step4_std = float(step4_stability_stats.loc[metric_name, "std"])
        print(
            f"{metric_name:<10}: "
            f"step3 mean={step3_mean:.4f}, std={step3_std:.4f} | "
            f"step4 mean={step4_mean:.4f}, std={step4_std:.4f}, std delta={step4_std - step3_std:+.4f}"
        )

    # Interpretation keeps focus on "what changed" and "why" for learning.
    print("\nInterpretation:")
    print("L1 regularization can push some coefficients exactly to zero, creating a simpler model.")
    print("This helps feature selection intuition: only some features keep influence.")
    print("Step 5 will blend L1 and L2 using Elastic Net.")

    print("\nTop 10 features by |coefficient| in Step 4 model:")
    coefficient_frame = pd.DataFrame(
        {"feature": features.columns, "coefficient": step4_coef, "abs_coef": np.abs(step4_coef)}
    ).sort_values("abs_coef", ascending=False)
    for _, row in coefficient_frame.head(10).iterrows():
        print(f"{row['feature']:<30} coef={row['coefficient']:+.4f}")

    plot_step4_summary(
        y_test=step4_result["y_test"],
        y_pred=step4_result["y_pred"],
        y_proba=step4_result["y_proba"],
        feature_names=list(features.columns),
        coefficients=step4_coef,
        output_path=args.output_path,
    )
    print(f"\nSaved figure to: {args.output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
