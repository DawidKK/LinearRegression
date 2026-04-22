.PHONY: sync lint test train train-poly train-multi train-multi-scaled train-ridge-sweep train-lasso-sweep train-elastic-net-cv train-logistic-step1 train-logistic-step2 train-logistic-step3 train-logistic-step4 train-logistic-step5 train-logistic-step6 traing-logistic-regression-feature-scalling traing-logistic-regression-l2-regularization-c-tuning traing-logistic-regression-l1-lasso traing-logistic-regression-elastic-net traing-logistic-regression-threshold-tuning notebook format

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix

test:
	uv run pytest -q

train:
	uv run python scripts/linear_regression/train_baseline.py

train-poly:
	uv run python scripts/linear_regression/train_housing_polynomial.py

train-multi:
	uv run python scripts/linear_regression/train_housing_multifeature_linear.py

train-multi-scaled:
	uv run python scripts/linear_regression/train_housing_multifeature_scaled_linear.py

notebook:
	uv run jupyter lab

train-ridge-sweep:
	uv run python scripts/linear_regression/train_housing_ridge_alpha_sweep.py

train-lasso-sweep:
	uv run python scripts/linear_regression/train_housing_lasso_alpha_sweep.py

train-elastic-net-cv:
	uv run python scripts/linear_regression/train_housing_elastic_net_cv.py

train-logistic-step1:
	uv run python scripts/logistic_regression/step1_raw_logistic_regression_baseline.py

train-logistic-step2:
	uv run python scripts/logistic_regression/step2_logistic_regression_with_scaling.py

train-logistic-step3:
	uv run python scripts/logistic_regression/step3_logistic_regression_l2_c_tuning.py

train-logistic-step4:
	uv run python scripts/logistic_regression/step4_logistic_regression_l1_lasso.py

train-logistic-step5:
	uv run python scripts/logistic_regression/step5_logistic_regression_elastic_net.py

train-logistic-step6:
	uv run python scripts/logistic_regression/step6_logistic_regression_threshold_tuning.py

traing-logistic-regression-threshold-tuning:
	uv run python scripts/logistic_regression/step6_logistic_regression_threshold_tuning.py
