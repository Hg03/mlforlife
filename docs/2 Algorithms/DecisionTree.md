---
icon: lucide/vault
---

# 4. Decision Trees

Decision Trees are one of the most fundamental yet versatile algorithms in a machine learning toolkit. They are simple enough to explain to non-technical stakeholders, yet powerful enough to serve as the building blocks for state-of-the-art models like **Random Forests** and **XGBoost**.

At its heart, a decision tree breaks a complex problem into smaller, simpler ones by asking a series of **yes/no questions**.

---

## Core Concept: How They Work

Each internal node in a tree represents a **decision rule**, each branch represents a **possible outcome**, and each leaf node represents a **final prediction**.

Unlike linear models, decision trees:

* Model **nonlinear relationships** naturally.
* Capture **feature interactions** (e.g., "If Age > 30 AND Income > 70k").
* Are **non-parametric**, meaning they don't assume a specific distribution (like a Normal distribution) for your data.

---

## Assumptions of Decision Trees

While trees are flexible, they rely on a few implicit principles:

| Assumption | Description |
| --- | --- |
| **Feature Relevance** | Features must contain signal; trees can struggle if the data is mostly noise. |
| **Axis-Aligned Splits** | Splits happen on one feature at a time, creating "staircase" boundaries. |
| **i.i.d. Samples** | Data points should be independent and identically distributed. |
| **Threshold-Based** | Relationships are best captured through "greater than/less than" conditions. |

---

## How a Tree is Built (Optimization)

The goal is to find splits that reduce **impurity**—making the resulting child nodes as "pure" as possible.

### 1. Splitting Criteria

* **Entropy & Information Gain:** Measures disorder. We pick the split that maximizes the reduction in entropy.

$$Entropy(S) = - \sum p_i \log_2(p_i)$$


* **Gini Impurity:** Used by the CART algorithm. It measures the probability of misclassification.

$$Gini = 1 - \sum p_i^2$$


* **Variance Reduction:** Used for **Regression Trees** to minimize the spread of continuous values in each leaf.

### 2. Preventing Overfitting

A tree that grows too deep will memorize the training data (High Variance). We control this through:

* **Pre-pruning (Early Stopping):** Setting `max_depth`, `min_samples_leaf`, or `max_leaf_nodes`.
* **Post-pruning (Cost Complexity Pruning):** Growing a full tree and then removing branches that provide little predictive power relative to their complexity.

---

## Evaluation Metrics

### For Classification

* **Accuracy:** Best for balanced datasets.
* **Precision/Recall/F1-Score:** Crucial for **imbalanced data** (e.g., fraud detection).
* **ROC-AUC:** Measures the model's ability to distinguish between classes across different thresholds.

### For Regression

* **MSE (Mean Squared Error):** Penalizes large errors heavily.
* **MAE (Mean Absolute Error):** More robust to outliers.
* **$R^2$ Score:** Measures the proportion of variance explained by the model.

---

## Practical Tips and Pitfalls

* **Scale Invariance:** One of the biggest perks! Trees do **not** require feature scaling (standardization or normalization).
* **Categorical Encoding:** Trees cannot handle strings directly. Use **One-Hot Encoding** for nominal data or **Ordinal Encoding** for ordered data.
* **The Stability Issue:** Single trees are unstable—small data changes can flip the entire tree structure.
* *Solution:* Use **Ensembles** (Random Forests or Boosting) to average out this variance.


* **Imbalanced Data:** Trees are biased toward the majority class. Use `class_weight='balanced'` or resampling techniques like **SMOTE**.
* **Feature Importance:** Trees naturally rank features based on how much they reduce impurity. This is a powerful tool for feature selection.

### [Quick PDF Version]()