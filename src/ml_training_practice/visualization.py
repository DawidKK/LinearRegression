from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .modeling import HousingLinearRegressionResult, HousingPolynomialRegressionResult


def plot_housing_linear_regression_result(
    result: HousingLinearRegressionResult,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(figsize=(10, 6))

    # Plot train and test points separately so you can visually inspect the split.
    axis.scatter(
        result.x_train["year_built"],
        result.y_train,
        label="Train data",
        color="#1f77b4",
        alpha=0.8,
    )
    axis.scatter(
        result.x_test["year_built"],
        result.y_test,
        label="Test data",
        color="#ff7f0e",
        alpha=0.9,
    )

    all_sqft = np.concatenate(
        [result.x_train["year_built"].to_numpy(), result.x_test["year_built"].to_numpy()]
    )
    # Build a smooth x-axis range and run model predictions to draw the fitted straight line.
    line_x_values = np.linspace(all_sqft.min(), all_sqft.max(), 200)
    line_x = pd.DataFrame({"year_built": line_x_values})
    line_y = result.model.predict(line_x)
    axis.plot(line_x_values, line_y, color="#2ca02c", linewidth=2, label="Linear regression line")

    axis.set_title("Housing Prices: Linear Regression Curve")
    axis.set_xlabel("year_built")
    axis.set_ylabel("price_usd")
    axis.grid(alpha=0.25)
    axis.legend()

    metrics_text = f"RMSE: {result.rmse:,.2f}\nR²: {result.r2:.3f}"
    # Put model quality metrics on the chart for quick interpretation.
    axis.text(
        0.02,
        0.98,
        metrics_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        # Ensure output directory exists before saving the figure.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)

    return figure, axis


def plot_housing_polynomial_regression_comparison(
    results: list[HousingPolynomialRegressionResult],
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    if not results:
        raise ValueError("results must not be empty")

    figure, axis = plt.subplots(figsize=(10, 6))
    first_result = results[0]

    # Plot train and test points once; fitted curves vary by polynomial degree.
    axis.scatter(
        first_result.x_train["year_built"],
        first_result.y_train,
        label="Train data",
        color="#1f77b4",
        alpha=0.8,
    )
    axis.scatter(
        first_result.x_test["year_built"],
        first_result.y_test,
        label="Test data",
        color="#ff7f0e",
        alpha=0.9,
    )

    all_years = np.concatenate(
        [first_result.x_train["year_built"].to_numpy(), first_result.x_test["year_built"].to_numpy()]
    )
    line_x_values = np.linspace(all_years.min(), all_years.max(), 300)
    line_x = pd.DataFrame({"year_built": line_x_values})
    line_colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for color, result in zip(line_colors, results):
        line_x_polynomial = result.polynomial_features.transform(line_x)
        line_y = result.model.predict(line_x_polynomial)
        axis.plot(
            line_x_values,
            line_y,
            color=color,
            linewidth=2,
            label=f"Degree {result.degree} (R²={result.r2:.3f})",
        )

    axis.set_title("Housing Prices: Linear vs Polynomial Regression Fit")
    axis.set_xlabel("year_built")
    axis.set_ylabel("price_usd")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)

    return figure, axis
