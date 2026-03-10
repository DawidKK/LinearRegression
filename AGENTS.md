# AGENTS.md

## Project Context

This project is part of my machine learning learning journey.
I am learning ML from a **low-level perspective** and want to understand how algorithms work internally before using complex models.

I already implemented:

- Linear Regression (from scratch)
- Logistic Regression (from scratch)

Now I want to practice using models from **scikit-learn**, but still progress gradually from simple models to more complex ones.

---

## Learning Goals

The goal of this repository is **not just to get the best performance**, but to **build intuition about machine learning models**.

I want to:

1. Start with simple linear models
2. Understand how regularization works
3. Compare simple vs complex models
4. Observe where simple models fail
5. Gradually move to more powerful models

---

## Preferred Model Progression

Models should generally be introduced in this order:

1. Linear Regression
2. Ridge Regression (L2 regularization)
3. Lasso Regression (L1 regularization)
4. Elastic Net
5. Logistic Regression (for classification)
6. Decision Trees
7. Random Forest
8. Gradient Boosting
9. XGBoost / LightGBM (later)

Avoid jumping directly to complex models unless explicitly requested.

---

## Coding Preferences

When generating code:

- Keep implementations **simple and readable**
- Prefer **explicit steps over magic abstractions**
- Avoid overly complex pipelines unless needed
- Add comments explaining what is happening

I want to **understand the code**, not just run it.

---

## Experimentation Goals

For each model I want to practice:

- Train/test split
- Model training
- Model evaluation
- Hyperparameter tuning
- Comparing model performance

Important metrics to use:

Regression:

- RMSE
- R²

Classification:

- Accuracy
- Precision
- Recall
- ROC AUC

---

## Regularization Practice

When working with linear models, include experiments with:

- Ridge (L2 regularization)
- Lasso (L1 regularization)

Focus on understanding:

- the effect of the regularization parameter (`alpha`)
- how coefficients change
- when features are removed by Lasso

---

## Guidance for AI Assistance

When assisting in this project:

- Prefer **teaching-oriented explanations**
- Suggest experiments instead of only solutions
- Encourage comparison between models
- Avoid hiding important logic inside libraries

The goal is **learning machine learning**, not just producing results.
