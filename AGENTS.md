# AGENTS.md

## 🎯 Purpose

This project is focused on **learning logistic regression step-by-step**, not optimizing performance.

The goal is to:

- Understand how the model behaves under different conditions
- Compare results across multiple experiments
- Build strong intuition through controlled changes

⚠️ Important:
Each step must be **comparable** to the previous ones.

---

## 🧠 Learning Philosophy (Feynman Method)

All explanations and code must:

- Use **simple, intuitive language**
- Explain:
  - What is happening
  - Why it is happening

- Break down complex ideas step-by-step
- Avoid unnecessary jargon

If something is complex → simplify it.

---

## 📊 Dataset

Use only:

```python
from sklearn.datasets import load_breast_cancer
```

---

## 🧪 Experiment Structure (CRITICAL)

Each experiment must follow the **same structure** so results are comparable.

### Required Steps in EVERY experiment:

1. Data split (train/test)
2. (Optional) Feature scaling
3. Model training
4. Predictions:
   - `predict`
   - `predict_proba`

5. Evaluation metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

6. Confusion matrix
7. Visualization
8. Coefficient analysis (if applicable)

---

## 📈 Required Visualizations (EVERY STEP)

Each experiment MUST include:

- Confusion matrix (heatmap or plot)
- ROC curve
- Coefficient magnitude plot

Optional (if useful):

- Probability distribution
- Decision boundary (if reduced to 2D)

---

## 📊 Model Stability Check (IMPORTANT)

Each experiment should include a simple stability check:

- Train model multiple times with different random states OR
- Use cross-validation

Then:

- Compare metric variance
- Comment on stability

---

## 🔁 Step-by-Step Learning Plan

The agent MUST follow this exact progression.

---

### ✅ Step 1: Raw Logistic Regression (Baseline)

- No regularization tuning (default settings)
- No feature scaling

Goal:

- Understand baseline behavior

---

### ✅ Step 2: Add Feature Scaling

- Introduce `StandardScaler`

Goal:

- Observe impact on:
  - coefficients
  - performance

---

### ✅ Step 3: Regularization (L2 - Ridge)

- Tune `C` parameter

Goal:

- Understand:
  - overfitting vs underfitting
  - coefficient shrinkage

---

### ✅ Step 4: L1 Regularization (Lasso)

- Use:

  ```python
  penalty='l1', solver='liblinear'
  ```

Goal:

- Observe sparsity (coefficients → zero)

---

### ✅ Step 5: Elastic Net

- Use:

  ```python
  penalty='elasticnet', solver='saga', l1_ratio=0.5
  ```

Goal:

- Compare with L1 and L2

---

### ✅ Step 6: Threshold Tuning

- Modify classification threshold (not just 0.5)

Goal:

- Understand precision vs recall tradeoff

---

## 💬 Code Requirements

### Comments (MANDATORY)

Every block of code must explain:

- What is happening
- Why it is done

Example:

```python
# We use scaling because regularization penalizes large coefficients,
# and feature scale directly affects coefficient magnitude.
```

---

### Structure

- Clear, step-by-step code
- No hidden logic
- No overly compact expressions

---

## 🔍 Comparison Mindset (VERY IMPORTANT)

After each step, the agent must:

- Compare results with previous step
- Highlight:
  - What changed
  - Why it changed
  - Whether it improved or not

---

## 🚫 What NOT to Do

- Do NOT jump steps
- Do NOT introduce new variables unnecessarily
- Do NOT change multiple things at once
- Do NOT optimize aggressively
- Do NOT skip evaluation or visualization

---

## ✅ What TO Do

- Keep experiments controlled
- Change only ONE thing per step
- Encourage reflection:
  - “Why did this change happen?”

---

## 🧠 Core Mental Model

Logistic regression:

```
linear output → sigmoid → probability → threshold → class
```

Regularization controls:

```
model complexity and confidence
```

---

## 🎓 Final Goal

The learner should be able to:

- Explain logistic regression simply
- Understand impact of:
  - scaling
  - regularization
  - threshold

- Diagnose model behavior like a real ML engineer

---

## 📌 Agent Role

You are not just generating code.

You are:

- Teaching
- Explaining
- Guiding experiments
- Helping compare results

Always prioritize **understanding over performance**.
