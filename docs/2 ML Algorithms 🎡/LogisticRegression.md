---
icon: lucide/vault
---

# 2. Logistic Regression

If you’ve ever sat in a data science interview, chances are you’ve been asked about **Logistic Regression**. Despite being one of the oldest algorithms in the machine learning toolkit, it continues to be one of the most important—and one of the most misunderstood.

While Linear Regression predicts continuous values, Logistic Regression predicts **probabilities**, helping us decide whether an email is spam, a transaction is fraudulent, or a customer is likely to churn. This simple switch makes Logistic Regression the gateway to understanding **classification**.

---

## What is Logistic Regression?

At its core, **Logistic Regression** is a classification algorithm used to predict **discrete outcomes**—most commonly, a binary one (0 or 1).

### From Linear Regression to Classification

Linear Regression fits a straight line ($y = \beta_0 + \beta_1x$) where the output can be anything from negative to positive infinity. However, probabilities must lie between **0 and 1**.

Logistic Regression fixes this by transforming the linear output using the **Sigmoid (or Logistic) function**:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n$.

* If $p > 0.5$, we predict **class 1**.
* If $p \leq 0.5$, we predict **class 0**.

### The Log-Odds (The "Logit")

The relationship is modeled linearly in terms of **log-odds**:

$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n$$

This connects linear relationships with probabilistic reasoning, making the model highly **interpretable**.

---

## Assumptions of Logistic Regression

| Assumption | Description |
| --- | --- |
| **Binary Target** | The dependent variable must be categorical (usually binary). |
| **Independence** | Observations must be independent of each other (no repeated measures). |
| **Linearity of Log-Odds** | There must be a linear relationship between independent variables and the **log-odds**. |
| **No Multicollinearity** | Independent variables should not be highly correlated (checked via **VIF**). |
| **Sample Size** | Requires a sufficiently large sample (typically 10–15 observations per predictor). |
| **No Influential Outliers** | Extreme points can disproportionately shift the decision boundary. |

---

## How Logistic Regression is Optimized

Because the sigmoid function is non-linear, we cannot use Ordinary Least Squares. Instead, we use **Maximum Likelihood Estimation (MLE)**.

### 1. The Likelihood Function

The goal is to find the coefficients $\beta$ that maximize the probability of observing the given data. We minimize the **Negative Log-Likelihood**, also known as **Binary Cross-Entropy Loss (Log Loss)**:

$$J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)]$$

### 2. Gradient Descent

We use an iterative approach to find the minimum of the Log Loss function:


$$\beta_j := \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}$$

### 3. Regularization

To prevent overfitting, we add penalty terms:

* **L1 (Lasso):** Adds $\lambda \sum |\beta_j|$. Can drive coefficients to zero (Feature Selection).
* **L2 (Ridge):** Adds $\lambda \sum \beta_j^2$. Shrinks coefficients to handle multicollinearity.

---

## Common Evaluation Metrics

| Metric | Definition | When to Use |
| --- | --- | --- |
| **Accuracy** | $(TP+TN) / Total$ | Balanced datasets only. |
| **Precision** | $TP / (TP+FP)$ | When the cost of a **False Positive** is high (e.g., Spam detection). |
| **Recall** | $TP / (TP+FN)$ | When the cost of a **False Negative** is high (e.g., Cancer detection). |
| **F1-Score** | $2 \cdot \frac{Prec \cdot Rec}{Prec + Rec}$ | For imbalanced datasets; the harmonic mean of both. |
| **ROC-AUC** | Area Under the Curve | Measures how well the model **ranks** classes, regardless of threshold. |
| **Log Loss** | Binary Cross-Entropy | Measures how well-calibrated the probabilities are. |

---

## Practical Tips and Pitfalls

* **Feature Scaling:** Always standardize features (like Income vs. Age) when using **regularization** so the penalty is applied fairly.
* **The Dummy Variable Trap:** When encoding categorical variables, always use `drop_first=True` to avoid perfect multicollinearity.
* **Imbalanced Data:** Accuracy is a trap! Use **SMOTE**, adjust **class weights**, or focus on **Precision-Recall curves**.
* **Interpreting Coefficients:** Exponentiate the coefficients ($e^{\beta}$) to get the **Odds Ratio**. An odds ratio of 1.2 means a 1-unit increase in $X$ increases the odds of the outcome by 20%.
* **Calibration:** Use **Platt Scaling** or **Isotonic Regression** if your predicted probabilities don't match real-world frequencies.

### [Quick PDF Version](https://github.com/Hg03/mlforlife/blob/main/docs/pdfs/Mastering_Logistic_Regression.pdf)