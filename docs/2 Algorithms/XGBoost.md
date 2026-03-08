---
icon: lucide/vault
---

# 6. XGBoost

XGBoost (Extreme Gradient Boosting) has become the gold standard for machine learning on tabular data. It strikes a remarkable balance between **accuracy**, **speed**, and **practicality**, making it a favorite for both industry projects and Kaggle competitions.

---

## The Core Idea: Gradient Boosting

Gradient boosting builds a strong model by combining many "weak" decision trees. Unlike Random Forests, which build trees independently in parallel, boosting builds trees **sequentially**.

1. **Baseline:** Start with a simple guess (e.g., the average house price).
2. **Calculate Residuals:** Find the errors (actual price - predicted price).
3. **Predict Errors:** Train a new tree specifically to predict those residuals.
4. **Update:** Add this tree's prediction to the previous guess (scaled by a **learning rate**).
5. **Repeat:** Each new tree focuses on the mistakes made by the previous ensemble.

It’s called **Gradient** boosting because each new tree is essentially fitting the negative gradient of the loss function—mathematically "stepping" toward the minimum error.

---

## What Makes XGBoost "Extreme"?

While basic gradient boosting is powerful, XGBoost introduces several innovations that make it faster and more robust:

### 1. Regularization

XGBoost includes **L1 (Lasso)** and **L2 (Ridge)** regularization directly in its objective function. This penalizes complex trees, pushing leaf weights toward zero and preventing the model from over-fitting to noise.

### 2. Weighted Quantile Sketch

Finding the best split point for a feature with thousands of unique values is slow. XGBoost uses a "sketch" to propose a small number of candidate split points based on the distribution of the data, weighted by the gradients.

### 3. Sparsity-Aware Splitting (Handling Missing Values)

XGBoost handles missing data automatically. During training, it tries sending missing values to both the left and right child nodes and learns the "default" direction that minimizes the loss.

### 4. System Optimizations

* **Parallelization:** It evaluates features in parallel across CPU cores.
* **DMatrix:** A custom data structure that stores data in a compressed, column-oriented format for faster memory access.
* **Cache-Aware Access:** It optimizes how the CPU reads data to minimize "cache misses."

---

## How it Learns: Gradients and Hessians

XGBoost uses a Taylor expansion to approximate the loss function. For every data point, it calculates:

* **Gradient ($g$):** The first derivative (direction of the error).
* **Hessian ($h$):** The second derivative (the curvature/confidence of the error).

The model evaluates a split by summing these values. The **Gain** of a split is calculated as:


$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$


Where $\lambda$ is L2 regularization and $\gamma$ is the penalty for adding a new leaf.

---

## Important Parameters

| Parameter | Description | Impact |
| --- | --- | --- |
| `eta` (Learning Rate) | Scales the contribution of each tree. | Smaller values (0.01-0.1) require more trees but generalize better. |
| `max_depth` | Maximum depth of a tree. | Higher values increase complexity and risk of overfitting. |
| `min_child_weight` | Minimum sum of instance weight (Hessian) needed in a child. | Higher values make the model more conservative. |
| `gamma` | Minimum loss reduction required to make a split. | Acts as a complexity control; higher values = fewer splits. |
| `subsample` | Fraction of rows to sample for each tree. | Introduces randomness to prevent overfitting. |
| `lambda` / `alpha` | L2 and L1 regularization terms. | Prevents leaf weights from becoming too large. |

---

## Feature Importance

XGBoost offers three ways to rank features:

1. **Weight:** The number of times a feature is used to split the data.
2. **Gain:** The total reduction in loss brought by that feature. (Usually the most reliable).
3. **Cover:** The number of samples affected by splits using that feature.

---

## Practical Tuning Strategy

1. **Baseline:** Start with a high `learning_rate` (0.1) and find the optimal number of trees using **early stopping**.
2. **Tree Structure:** Tune `max_depth` and `min_child_weight`.
3. **Robustness:** Tune `gamma` to control pruning.
4. **Randomness:** Tune `subsample` and `colsample_bytree`.
5. **Refine:** Lower the `learning_rate` and increase the number of trees for final performance.