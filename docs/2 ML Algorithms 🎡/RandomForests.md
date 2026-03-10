---
icon: lucide/trees
---

# 5. Random Forests

If you have ever trained a decision tree and noticed that it performs very well on the training data but poorly on unseen data, you have already encountered the core weakness of trees: **overfitting**.

Random Forests were designed to solve this problem. The main idea is simple: instead of relying on a single decision tree, build many of them and let them collectively make the final decision. By averaging or voting across multiple trees, the model becomes more accurate and stable.

---

## The Core Idea: Ensemble Learning and Bagging

Random Forest is based on **Ensemble Learning**, which combines several models to produce a collective judgment. The specific technique used is **Bagging** (Bootstrap Aggregation).

### How Bagging Works:

1. **Bootstrap Sampling:** Create several new datasets by randomly sampling the original training data **with replacement**.
2. **Parallel Training:** Train a separate decision tree on each of these bootstrap samples.
3. **Aggregation:** Combine the predictions.
* **Classification:** Majority voting.
* **Regression:** Averaging the results.



Bagging reduces **variance** without increasing bias, making high-variance models like decision trees much more stable.

---

## Random Forest: A Smarter Ensemble

A Random Forest takes bagging a step further by introducing **Feature Randomness**.

In a standard bagged ensemble, if one feature is a very strong predictor, every tree will choose that feature for its first split. This makes the trees highly correlated. Random Forests solve this by:

* At each split, the tree only considers a **random subset of features**.
* This forces trees to look at different patterns, ensuring the ensemble is diverse.

---

## Out-of-Bag (OOB) Samples and Error

Because Random Forests use bootstrap sampling, about **one-third** of the data is left out of each tree's training set. These are called **Out-of-Bag (OOB) samples**.

Each tree can be tested on the OOB data it didn't see during training. By aggregating these "self-tests," the model provides an **OOB Error** estimate, which acts as a built-in validation score without needing a separate test set.

---

## Key Hyperparameters

| Hyperparameter | Role | Effect of Increasing |
| --- | --- | --- |
| `n_estimators` | Number of trees in the forest. | Reduces variance; stabilizes the model. |
| `max_depth` | Max depth of each individual tree. | Increases complexity; risks overfitting. |
| `max_features` | Number of features considered at each split. | Higher values make trees more similar; lower increases diversity. |
| `min_samples_leaf` | Min samples required in a terminal leaf. | Smoothes the model; reduces noise sensitivity. |
| `bootstrap` | Whether to use replacement when sampling. | Usually kept `True` for better generalization. |

---

## Understanding Feature Importance

Random Forests are highly interpretable because they reveal which variables drive predictions.

1. **Impurity-Based Importance (Gini Importance):** Measures how much each feature reduces impurity (Gini/Entropy) across all trees. It is fast but can be biased toward continuous features with many unique values.
2. **Permutation Importance:** Shuffles the values of a single feature and measures the drop in model accuracy. If the score drops significantly, the feature is highly important.

---

## Practical Interview Tips

* **Overfitting:** Explain that while individual trees overfit, the forest as a whole does not because the errors of diverse trees average out.
* **Scaling:** Mention that Random Forests (like Decision Trees) do **not** require feature scaling.
* **Correlation:** Be ready to explain that feature randomness is used specifically to **de-correlate** the trees, making the ensemble more robust.

### [Quick PDF Version](https://github.com/Hg03/mlforlife/blob/main/docs/pdfs/Mastering_Random_Forests.pdf)