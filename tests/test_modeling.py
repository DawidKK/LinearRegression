import matplotlib

from ml_training_practice import train_housing_univariate_linear_regression
from ml_training_practice.data import load_housing_prices_data
from ml_training_practice.visualization import plot_housing_linear_regression_result

matplotlib.use("Agg")


def test_load_housing_prices_data_has_expected_columns_and_rows() -> None:
    housing_data = load_housing_prices_data()

    expected_columns = {"year_built", "price_usd"}
    assert expected_columns.issubset(set(housing_data.columns))
    assert len(housing_data) > 0


def test_train_housing_univariate_linear_regression_returns_expected_artifacts() -> None:
    result = train_housing_univariate_linear_regression()

    assert hasattr(result.model, "predict")
    assert isinstance(result.rmse, float)
    assert isinstance(result.r2, float)
    assert len(result.test_predictions) == len(result.y_test)
    assert isinstance(result.coefficient, float)
    assert isinstance(result.intercept, float)


def test_plot_housing_linear_regression_result_returns_figure_and_saves_file(tmp_path) -> None:
    result = train_housing_univariate_linear_regression()
    output_path = tmp_path / "housing_linear_curve.png"

    figure, axis = plot_housing_linear_regression_result(result, save_path=output_path)

    assert figure is not None
    assert axis is not None
    assert output_path.exists()
