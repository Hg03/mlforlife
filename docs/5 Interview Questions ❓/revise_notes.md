---
icon: lucide/briefcase
---

## 3. AI/ML Engineer Interview Revision Notes
   
### Table of Contents
1. Machine Learning Fundamentals
2. Time Series Concepts
3. Demand Forecasting
4. Deep Learning Basics & Transformers
5. Tokens, Embeddings & Position Encodings
6. LLM Architecture & Inference Parameters
7. LLM Training Pipeline: Pretraining → SFT → RLHF/DPO
8. Prompt Engineering & Few-Shot Learning
9. Agents
10. Model Context Protocol (MCP)
11. Embeddings Deep-Dive
12. RAG (Retrieval-Augmented Generation)
13. Multimodal Models (Vision + Language)
14. Graph RAG & Knowledge Graphs
15. Distributed Training & Inference (surface-level, but expect it)
16. Observability & Evaluation
17. Failure Diagnosis Playbooks
18. Testing & Eval-Driven Development for LLM Apps
19. Coding / Whiteboard Practice
20. System Design for LLM/RAG Applications
21. Deploying Forecasting & RAG Solutions on AWS
22. Behavioral / Project Deep-Dive Prep

---

## 1. Machine Learning Fundamentals

### Core Terminology (quick-fire definitions)

| Term | Definition |
|---|---|
| **Feature** | Input variable used for prediction |
| **Label / Target** | The value being predicted |
| **Overfitting** | Model memorizes training data, fails on unseen data (low bias, high variance) |
| **Underfitting** | Model too simple, fails to capture patterns (high bias, low variance) |
| **Bias-Variance Tradeoff** | Bias = error from wrong assumptions; Variance = error from sensitivity to training data noise. Total error = Bias² + Variance + Irreducible error |
| **Regularization** | Penalizes model complexity to reduce overfitting (L1/Lasso, L2/Ridge, ElasticNet) |
| **Cross-validation** | Splitting data into k folds to validate model robustness (k-fold CV) |
| **Precision** | TP / (TP + FP) — "Of predicted positives, how many are correct" |
| **Recall** | TP / (TP + FN) — "Of actual positives, how many caught" |
| **F1 Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Area under TPR vs FPR curve — measures separability across thresholds |
| **Gradient Descent** | Iterative optimization: `θ = θ - α∇J(θ)` |
| **Learning Rate** | Step size in gradient descent |
| **Loss Function** | MSE (regression), Cross-Entropy (classification) |
| **Bagging** | Train multiple models on bootstrapped samples, average predictions (reduces variance) — e.g., Random Forest |
| **Boosting** | Sequentially train models, each correcting previous errors (reduces bias) — e.g., XGBoost, AdaBoost |
| **Curse of Dimensionality** | As features increase, data becomes sparse, distance metrics lose meaning |
| **Feature Scaling** | Normalization (0-1) or Standardization (mean 0, std 1) — needed for distance-based/gradient-based models |

---

### Top Algorithms

#### **Linear Regression**
- Predicts continuous output: `y = w·x + b`
- Loss: **Mean Squared Error (MSE)**
- Assumptions: linearity, independence, homoscedasticity (constant variance), normal residuals, no multicollinearity
- **Interview trap:** Multicollinearity → unstable coefficients. Fix: VIF check, drop correlated features, or use Ridge.

#### **Logistic Regression**
- Classification via **sigmoid**: `σ(z) = 1/(1+e^-z)`
- Loss: **Binary Cross-Entropy (Log Loss)**
- Outputs probability → threshold (default 0.5) → class
- Decision boundary is **linear**
- Multi-class: Softmax (multinomial logistic regression) or One-vs-Rest

#### **Support Vector Machine (SVM)**
- Finds hyperplane that **maximizes margin** between classes
- **Support vectors** = points closest to the boundary that define it
- **Kernel trick**: maps data to higher dimensions without explicit computation (RBF, polynomial, linear kernels) → handles non-linear boundaries
- **C parameter**: controls tradeoff between margin width and misclassification (low C = wider margin, more tolerance; high C = strict, less regularization)
- **Gamma (RBF)**: controls influence radius of a single point (high gamma = overfitting risk)
- Good for high-dimensional, small-to-medium datasets; struggles at scale

#### **Decision Tree**
- Splits data recursively using a feature/threshold that maximizes **information gain** or minimizes **Gini impurity**
- **Entropy**: `-Σ p(x)log2(p(x))` — measures disorder
- **Gini Impurity**: `1 - Σ p(x)²`
- Prone to overfitting → controlled via `max_depth`, `min_samples_split`, pruning
- Pros: interpretable, handles non-linear data, no scaling needed
- Cons: high variance, unstable (small data change → different tree)

#### **Random Forest**
- **Bagging** ensemble of decision trees
- Each tree trained on bootstrapped sample + random subset of features (**feature bagging**) → decorrelates trees
- Final prediction: majority vote (classification) / average (regression)
- Reduces variance vs single tree; robust to overfitting
- Provides **feature importance** naturally

#### **XGBoost (Extreme Gradient Boosting)**
- **Sequential boosting**: each tree corrects residual errors of previous ensemble
- Uses **second-order gradients (Hessian)** for optimization — faster convergence than plain gradient boosting
- Key features: regularization (L1/L2) built into objective, handles missing values natively, tree pruning via `max_depth` + `gamma` (min loss reduction to split), column/row subsampling
- **Why it wins competitions:** speed (parallelized split-finding), regularization, handles sparse data
- Hyperparameters to know: `learning_rate (eta)`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, `lambda/alpha` (regularization)
- LightGBM/CatBoost are common comparisons — LightGBM uses leaf-wise growth (faster, riskier overfit); CatBoost handles categorical features natively

**Quick comparison table:**

<img width="1024" height="559" alt="image" src="https://gist.github.com/user-attachments/assets/3a58af93-2403-4002-b8f0-ede2f281830f" />

---

## 2. Time Series Concepts

*Foundational for demand forecasting (Section 3) and for any "explain classical vs ML forecasting" interview question.*

### Core Terminology

| Term | Definition |
|---|---|
| **Trend** | Long-term upward/downward movement in the series |
| **Seasonality** | Regular, fixed-period pattern (e.g., weekly, yearly) |
| **Cyclic pattern** | Fluctuations without fixed period (e.g., business cycles) — differs from seasonality, which has a fixed known period |
| **Residual / Noise** | What's left after removing trend + seasonality — ideally close to white noise |
| **Stationarity** | Statistical properties (mean, variance, autocovariance) don't change over time |
| **Lag** | Value of the series at a previous time step (t-1, t-7, etc.) |
| **White noise** | Series with no autocorrelation, constant mean/variance — "unpredictable" component |
| **Random walk** | `y_t = y_{t-1} + ε_t` — next value is previous value plus noise; classic non-stationary series |
| **Autocorrelation** | Correlation of a series with a lagged version of itself |

### Stationarity — Why It Matters
- Most classical forecasting models (ARIMA family) **assume stationarity** — a non-stationary series (trending mean, changing variance) breaks the model's assumptions and produces unreliable forecasts
- **Tests**: 
  - **ADF (Augmented Dickey-Fuller)**: null hypothesis = series has a unit root (non-stationary); low p-value → reject null → stationary
  - **KPSS**: opposite null hypothesis (series IS stationary) — often used alongside ADF to cross-check
- **Fixes for non-stationarity**: differencing (`y_t - y_{t-1}`), log transform (stabilizes variance), seasonal differencing (`y_t - y_{t-m}`)

### Decomposition
- **Additive model**: `Y = Trend + Seasonality + Residual` — use when seasonal fluctuations are roughly constant in magnitude regardless of trend level
- **Multiplicative model**: `Y = Trend × Seasonality × Residual` — use when seasonal fluctuations grow/shrink proportionally with the trend level (common in retail/demand data)
- **Methods**: classical decomposition (simple moving-average based), **STL (Seasonal-Trend decomposition using Loess)** — more robust to outliers and handles seasonality that changes slowly over time, generally preferred over classical decomposition in practice

### ACF & PACF (model identification)
- **ACF (Autocorrelation Function)**: correlation between the series and its lag, including indirect effects passed through intermediate lags
- **PACF (Partial Autocorrelation Function)**: correlation between the series and its lag **after removing the effect of intermediate lags** — isolates the direct relationship
- **Usage for classical model order selection**:
  - PACF cuts off sharply after lag p → suggests **AR(p)**
  - ACF cuts off sharply after lag q → suggests **MA(q)**
  - Both decay gradually → suggests an ARMA (mixed) process

### ARIMA / SARIMA / SARIMAX
- **ARIMA(p, d, q)**:
  - **p**: AR order — regression on the series' own past values
  - **d**: differencing order — number of times differenced to achieve stationarity
  - **q**: MA order — regression on past forecast errors
- **SARIMA(p,d,q)(P,D,Q,m)**: adds a seasonal component with its own AR/I/MA orders and seasonal period `m` (e.g., m=12 for monthly data with yearly seasonality)
- **SARIMAX**: SARIMA + exogenous regressors (e.g., price, promotions, holidays, weather) — very relevant for demand forecasting since demand is rarely driven by history alone

### Exponential Smoothing Family
| Method | Handles |
|---|---|
| **SES (Simple Exponential Smoothing)** | Level only — no trend, no seasonality |
| **Holt's Linear Trend** | Level + trend |
| **Holt-Winters** | Level + trend + seasonality (additive or multiplicative) |

- Weighted average of past observations with exponentially decaying weights — recent observations matter more
- Simple, fast, surprisingly strong baseline, especially at short horizons

### Prophet (Meta)
- Additive decomposition model: `y(t) = trend(t) + seasonality(t) + holidays(t) + ε`
- Seasonality modeled via **Fourier terms**, trend via piecewise linear/logistic growth with automatic changepoint detection
- **Strengths**: handles missing data and outliers gracefully, easy to add holiday effects, minimal tuning needed, good default baseline
- **Weaknesses**: often underperforms a well-tuned ARIMA or ML model on accuracy; less suited for very high-frequency or highly irregular series

### ML & Deep Learning Approaches
- **Feature-based ML (XGBoost/LightGBM)**: reframe forecasting as tabular regression using lag features, rolling statistics, calendar features — very strong and popular in industry/competitions (e.g., M5 competition winners)
- **RNN/LSTM/GRU**: sequence models that learn temporal patterns directly; historically strong but increasingly outperformed by transformer-based or global models
- **DeepAR (Amazon)**: autoregressive RNN trained **globally across many related time series**, produces **probabilistic** forecasts (full predictive distribution, not just a point estimate) — foundational for Amazon Forecast
- **Temporal Fusion Transformer (TFT)**: attention-based, handles static covariates, known future inputs (e.g., planned promotions), and past observed inputs jointly; provides interpretable attention weights
- **N-BEATS / N-HiTS**: pure deep learning architectures (no recurrence/attention needed) that perform strongly on univariate forecasting benchmarks
- **PatchTST / Informer**: transformer variants optimized for long-horizon forecasting efficiency

### Model Selection Cheat Sheet: ARIMA vs LSTM vs Prophet

*A fast, defensible answer to "which model would you pick and why" — pick based on data characteristics first, then confirm with evaluation scores, not the other way around.*

| | **ARIMA (/SARIMA)** | **LSTM** | **Prophet** |
|---|---|---|---|
| **Best data fit** | Stationary (or differenced-to-stationary), linear relationships | Long-term dependencies, large datasets, non-linear/high-variability patterns, multiple/multivariate inputs | Clear seasonality, missing data, irregular time intervals, holiday/event effects |
| **Forecast horizon** | Short-term — accuracy degrades as the horizon grows | Can model long-range dependencies given enough data | Good general-purpose horizon, especially with strong seasonal structure |
| **Seasonality** | Limited (needs SARIMA extension) | Learns it implicitly if present in training data | Explicit first-class support (Fourier terms + holiday regressors) |
| **Data volume needed** | Works with fairly small series | Needs large datasets to train well — underperforms ARIMA/Prophet on short series | Robust even with gaps/missing data, doesn't need huge volume |
| **Interpretability** | High (explicit AR/I/MA coefficients) | Low (black box) | Medium (decomposed trend/seasonality/holiday components are inspectable) |
| **Typical use case** | Stable, linear series — e.g. short-term inventory levels, macro series with mild trend | Stock prices, demand with complex non-linear/volatile patterns, multivariate series (price, weather, promos as extra inputs) | Retail/e-commerce sales with weekly/yearly seasonality, web traffic, anything with holiday spikes |

**Decision heuristic:**
- Data is stationary + linear + short horizon → **ARIMA/SARIMA**
- Data is highly non-linear, volatile, or has multiple correlated input signals and you have enough history → **LSTM**
- Data has strong seasonality, missing/irregular observations, or holiday effects and you want a low-maintenance baseline → **Prophet**
- Overlapping characteristics (e.g., seasonal **and** non-linear/volatile) → consider a **hybrid/stacked** approach, such as LSTM for the non-linear/long-term signal combined with ARIMA for short-term residual correction, or an ensemble of Prophet (seasonality/holidays) + a residual model for the non-linear leftover.

**Practical workflow (don't skip this):** always run EDA first — check stationarity (ADF/KPSS), seasonality strength, missingness, and data volume — before committing to a model family. Different series (even within the same domain, e.g. different currency pairs or different SKUs) can behave very differently, so the right move is usually to shortlist 2-3 candidate models based on the data profile above, then let held-out evaluation (walk-forward, never random k-fold) make the final call rather than picking a favorite model up front.

**Evaluation caveat — R² on time series:** R² tends to look unusually low for time series models even when the model is doing well. This is because time series data is highly autocorrelated, so a naive baseline (e.g., "predict the last observed value") already explains a large share of the variance, leaving little additional variance for the model to capture. Don't be alarmed by a low R² in isolation — compare against a naive/seasonal-naive baseline (this is exactly what **MASE** does, see below) rather than judging R² on its own.

**Interview soundbite:** *"I don't pick a time series model by intuition — ARIMA fits stationary, linear, short-horizon data; LSTM fits large, non-linear, multivariate series with long-range dependencies; Prophet fits seasonal data with missing values or holiday effects. I confirm the choice with walk-forward evaluation against a naive baseline, and I'll stack models — e.g., Prophet for seasonality plus a residual model for non-linear leftover error — when the data shows overlapping characteristics."*

### Time-Series-Specific Cross-Validation
- **Never use random k-fold** — it leaks future information into training (a huge, common mistake)
- **Walk-forward / rolling-origin validation**: train on data up to time T, validate on T+1...T+h, then slide the origin forward and repeat
- **Expanding window**: training set grows each fold (uses all available history)
- **Sliding window**: training set is a fixed-size window that moves forward (useful when older data is less relevant, e.g., due to concept drift)

### Evaluation Metrics
| Metric | Notes |
|---|---|
| **MAE** | Mean Absolute Error — simple, same units as target |
| **RMSE** | Penalizes large errors more (squared term) |
| **MAPE** | Mean Absolute Percentage Error — intuitive but undefined/unstable near zero actuals, asymmetric (penalizes over-forecast less than under-forecast) |
| **sMAPE** | Symmetric MAPE — attempts to fix MAPE's asymmetry, still has edge cases near zero |
| **WAPE (Weighted APE)** | Aggregates errors and actuals before dividing — much more stable than MAPE for sparse/low-volume series, very common in demand forecasting |
| **MASE (Mean Absolute Scaled Error)** | Scale-free — compares model error to a naive (seasonal) forecast's error; MASE < 1 means you're beating the naive baseline |
| **R²** | Usually low for time series relative to other regression problems — autocorrelated data means a naive last-value baseline already explains much of the variance, so don't treat a low R² alone as a red flag |

**Interview trap:** MAPE breaks down badly on intermittent/low-volume demand data (division by near-zero actuals) — WAPE or MASE is the safer default for demand forecasting eval.

### Multi-Horizon Forecasting Strategies
- **Direct**: train a separate model per forecast horizon (h=1, h=2, ... h=n) — no error accumulation, but more models to maintain
- **Recursive**: forecast h=1, feed it back as input to forecast h=2, and so on — simple, but errors compound over the horizon
- **Multi-output**: single model predicts the entire horizon vector at once (common in DL models like TFT, N-BEATS) — captures cross-horizon dependencies directly

---

## 3. Demand Forecasting

*Applies Section 2's time series toolbox to the specific, very common business problem of predicting product/SKU-level demand for inventory and supply chain decisions.*

### Why Demand Forecasting Is Different From Generic Time Series
- You're rarely forecasting **one** series — usually thousands to millions of SKU × location combinations
- Forecasts feed directly into **business decisions with asymmetric costs** (stockout = lost sales/customer trust; overstock = holding cost, waste, markdowns) — accuracy alone isn't the whole story, forecast **bias** matters too
- Heavy external drivers: promotions, pricing, seasonality, weather, competitor actions, macroeconomic factors

### Key Challenges

| Challenge | Why it's hard | Common fix |
|---|---|---|
| **Intermittent / sparse demand** | Many periods with zero demand (e.g., slow-moving SKUs) — MAPE/RMSE misleading, classical ARIMA struggles | **Croston's method**, **TSB (Teunter-Syntetos-Babai)**, or global ML models that pool information across SKUs |
| **New product / cold start** | No historical data at all | Use analogous/similar product history, hierarchical pooling, attribute-based (content) features instead of pure history |
| **Promotions & price changes** | Sudden demand spikes not explainable by history alone | Include promo/price as **exogenous regressors** (SARIMAX, feature-based ML, TFT known-future-inputs) |
| **Hierarchical consistency** | Forecasts at SKU level must roll up consistently to category/region/total level | **Reconciliation**: top-down, bottom-up, middle-out, or **optimal reconciliation (MinT)** which adjusts all levels for statistical coherence |
| **Scale (thousands of series)** | Fitting individual ARIMA per SKU doesn't scale operationally | **Global models** — one model trained across all series (with SKU/store as a categorical/embedding feature), shares learned patterns, better cold-start behavior |

### Modeling Approaches in Practice

- **Classical, per-series**: ARIMA/SARIMA/Prophet fit independently per SKU — interpretable, fine for a small number of high-value series, doesn't scale well and can't share information across related products
- **Global feature-based ML**: single LightGBM/XGBoost model trained across all SKU-store combinations with lag features + calendar + price/promo + SKU/store as categorical — currently one of the most common industry-strength approaches (this is essentially what won the M5 forecasting competition)
- **Global deep learning**: DeepAR, TFT, N-BEATS trained across the full panel of series — best when there's enough data volume to benefit from shared representation learning, and when probabilistic output is needed
- **Amazon Forecast (managed AWS service)**: AutoML wrapper that tries multiple algorithms (ARIMA, ETS, Prophet, DeepAR+, CNN-QR) and picks/ensembles the best per use case — removes most of the algorithm-selection burden (see Section 21 for deployment details)

### Feature Engineering for Demand Forecasting
- **Calendar features**: day-of-week, month, is_holiday, is_weekend, days-to-next-holiday
- **Lag features**: t-1, t-7, t-14, t-28, t-365 (capture weekly/yearly seasonality)
- **Rolling statistics**: rolling mean/std/min/max over multiple windows (7d, 28d, 90d)
- **Price/promotion features**: current price, discount %, promo flag, days since last promo
- **External regressors**: weather, local events, competitor pricing (when available)
- **Entity features**: SKU category, brand, store cluster, or learned embeddings for high-cardinality IDs

### Probabilistic Forecasting (critical for inventory decisions)
- A single point forecast isn't enough to set safety stock — you need to know the **distribution** of likely demand
- Models output **quantile forecasts** (e.g., P10, P50, P90) trained via **pinball/quantile loss**
- Business use: set reorder points/safety stock based on a target **service level** (e.g., stock to the P90 forecast to achieve ~90% fill rate), rather than just the mean forecast

### Evaluation — Business-Aligned Metrics
- **WAPE** is the industry default for demand forecasting (robust to zero/low-volume SKUs, aggregates naturally across a portfolio)
- **Bias metrics** (mean error, not absolute) matter separately from accuracy — a model that's "accurate on average" but systematically under-forecasts will still cause chronic stockouts
- **Business-facing metrics**: fill rate / service level achieved, inventory holding cost vs. stockout cost tradeoff, forecast value added (FVA) — is the model actually beating a naive/manual forecast?

**Interview soundbite:** *"In demand forecasting, WAPE beats MAPE because MAPE blows up on near-zero actuals, which are common with slow-moving SKUs — WAPE aggregates errors and actuals before dividing, so it stays stable at portfolio scale."*

**Interview soundbite:** *"Point forecasts aren't enough for inventory decisions — you need quantile/probabilistic forecasts to translate a forecast into a safety-stock and service-level decision."*

---

## 4. Deep Learning Basics & Transformers

### Backpropagation (intuition, not derivation)
- Forward pass computes prediction → loss
- Backward pass uses **chain rule** to compute gradient of loss w.r.t. every weight, layer by layer, from output back to input
- Each weight update: `w = w - α * ∂L/∂w`
- Interview framing: "backprop is just efficient chain-rule bookkeeping via a computation graph"

<img width="1024" height="559" alt="image" src="https://gist.github.com/user-attachments/assets/ce6c5e16-72be-4be1-897d-bde1a2f35010" />


### Vanishing / Exploding Gradients
- **Vanishing**: gradients shrink exponentially through many layers (common with sigmoid/tanh, deep RNNs) → early layers stop learning
- **Exploding**: gradients grow unbounded → unstable updates, NaNs
- Fixes: ReLU family activations, residual/skip connections, gradient clipping, careful initialization (Xavier/He), normalization layers

<img width="1024" height="559" alt="image" src="https://gist.github.com/user-attachments/assets/ab00fa3a-d11e-4dc8-b1d3-28d8d51a4c1b" />


### Normalization
- **BatchNorm**: normalizes across the batch dimension per feature — great for CNNs, but breaks down with small batches / variable-length sequences
- **LayerNorm**: normalizes across the feature dimension per sample — used in transformers, batch-size independent
- **RMSNorm**: simplified LayerNorm (no mean-centering, only rescales by root-mean-square) — cheaper, used in LLaMA-family models

### Why Transformers Replaced RNNs/LSTMs
- **RNNs/LSTMs**: process sequentially (token-by-token) → can't parallelize across time steps → slow to train, and still struggle with very long-range dependencies (vanishing gradient over long sequences even with gating)
- **Transformers**: self-attention lets every token directly attend to every other token in **one step** → fully parallelizable across the sequence during training, and captures long-range dependencies directly (no signal decay over distance)
- Tradeoff transformers introduce: O(n²) attention cost vs RNN's O(n) — this is precisely the problem KV cache/PagedAttention/FlashAttention address at inference time

### CNN vs RNN vs Transformer (one-liner each)
- **CNN**: local receptive fields, great for spatial/grid data (images), weight sharing via convolution kernels
- **RNN/LSTM**: sequential, maintains hidden state, natural fit for sequences but slow + long-range issues
- **Transformer**: parallel, attention-based, dominant for language (and increasingly vision — ViT) due to scalability

### Transformer Architecture (Deep Dive)
*Introduced in "Attention Is All You Need" (2017) — the absolute cornerstone of modern NLP.*

**Core components:**
- **Self-Attention**: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
  - Q (Query), K (Key), V (Value) are linear projections of input
  - Allows model to dynamically look at other tokens to understand context (e.g., "it" → resolves to "animal" not "street")
  - Why divide by √d_k? Keeps dot-product magnitude stable, prevents softmax from saturating into near one-hot (vanishing gradients)

- **Multi-Head Attention**: multiple attention computations in parallel, concatenated → captures different relationship types (semantic, syntactic, long-range, etc.)

- **Positional Encoding**: injects order info (critical since Transformers process in parallel)
  - **Original (sinusoidal)**: uses sin/cos functions at different frequencies
  - **RoPE (Rotary Position Embedding)**: encodes relative position via rotation matrices, better extrapolation, used in LLaMA/GPT-NeoX
  - **ALiBi (Attention with Linear Biases)**: adds position-dependent bias to attention scores, simpler than RoPE, used in some newer models.

- **Feed Forward Network (FFN)**: per-token MLP after attention; typically two linear layers with ReLU/GELU

- **LayerNorm/RMSNorm** + **residual connections**: stabilize training, enable deeper architectures

**Variants:**
- **Decoder-only** (GPT-style): causal mask, autoregressive generation — dominant for LLMs (can't attend to future tokens)
- **Encoder-only** (BERT-style): bidirectional attention, good for classification/embeddings
- **Encoder-Decoder** (T5-style): good for seq2seq (translation, summarization)

### Why Attention is Expensive
- Complexity: **O(n²)** in sequence length (each token attends to every other token)
- Memory: **O(n²)** for attention weights matrix
- This is why long-context is hard and optimizations (KV cache, FlashAttention, sparse attention) matter critically

<img width="1024" height="559" alt="image" src="https://gist.github.com/user-attachments/assets/9c903de3-7874-4cab-a329-3d541bbd4c6e" />


---

## 5. Tokens, Embeddings & Position Encodings

### Tokenization
- **What it does**: converts raw text → sequence of integers (token IDs) that the model can process
- **Why it matters**: LLMs don't understand text; they understand numbers. Tokenization is the critical bridge
- **Common methods**: 
  - **BPE (Byte Pair Encoding)**: merges most frequent byte pairs iteratively (used in GPT models)
  - **WordPiece**: similar to BPE, used in BERT
  - **SentencePiece**: handles subword segmentation across languages, used in T5/mBART
- **Key gotcha**: same text tokenizes differently across models → affects input length estimates, model behavior, and cost calculations
- **Example gotcha**: punctuation handling (does `—` split into multiple tokens?), Unicode normalization, special tokens

**Interview trap:** Tokenizer mismatch between training and inference = silent failure. Always verify you're using the same tokenizer both ways.

### Embeddings
- **Definition**: dense vector representation of a token that captures semantic meaning
- **Process**: token ID (e.g., 42) → lookup in embedding matrix (d_model dimensions) → vector (e.g., 768-dim)
- **Property**: tokens with similar meanings cluster together in vector space (learned during pretraining)
- **Dimensions**: typical range 512-1536 (older BERT was 768, GPT-3 was 12288, GPT-4 is proprietary)
- Embedding matrix is **huge** (vocab_size × d_model) — often the largest single component of smaller models

### Position Encodings (detailed)
**Problem solved:** Transformers process all tokens in parallel, so they have no inherent sense of "order." Position encodings restore this.

**Methods:**
- **Sinusoidal (original Transformer)**: uses sin/cos at different frequencies — analytic, no learning needed, good extrapolation to longer sequences
- **Learnable embeddings**: train position embeddings like any other parameter — simpler but poor extrapolation beyond training length
- **Rotary (RoPE)**: applies rotation matrix based on position — encodes *relative* position (not absolute), excellent extrapolation, used in LLaMA/Mistral
- **ALiBi (Attention with Linear Biases)**: doesn't add to embeddings, instead subtracts a bias from attention scores based on distance — simplest, some evidence of better extrapolation

**Interview angle:** RoPE's relative-position encoding is why LLaMA models extrapolate well to longer contexts — this is a known win and worth mentioning if asked about long-context models.

---

## 6. LLM Architecture & Inference Parameters

### Context Window (Critical Constraint)
- **What it is**: max number of tokens the model can process at once (e.g., 8K, 32K, 128K, 200K)
- **Why it matters**: anything outside is completely ignored — no "overflow" behavior
- **Production implication**: RAG/prompt engineering must fit everything (instructions + context + query) within this budget
- **Current landscape**: 
  - GPT-3.5: 4K/16K
  - GPT-4: 8K/128K
  - Claude 3: 100K-200K (depends on variant)
  - Llama 3: 8K native (can be extended via RoPE scaling)

### KV Cache (Inference Efficiency)
- **Problem**: during autoregressive generation, each new token needs attention over all previous tokens
- **Naive approach**: recompute K, V for entire sequence at every new token → wasteful
- **KV Cache solution**: store computed Key & Value tensors from previous tokens, reuse them, only compute K/V for the new token
- **Tradeoff**: **Memory for speed** — KV cache grows linearly with (sequence_length × batch_size × num_layers × num_heads × head_dim)
- **Real impact**: KV cache is often the **memory bottleneck** in serving LLMs, not model weights
- **Cost at scale**: for 70B model serving 1000-token context at batch 32, KV cache ≈ 40GB (vs model weights ≈ 140GB)

### PagedAttention (vLLM's innovation)
- **Problem**: naive KV cache allocation reserves **contiguous memory** for max sequence length per request → massive fragmentation when sequences vary in length
- **Solution** (inspired by OS virtual memory): 
  - Split KV cache into fixed-size **blocks** ("pages", e.g., 16 tokens each)
  - Store non-contiguously, map via a block table (like page table in OS)
- **Benefits**:
  - Near-zero memory waste (allocate only what's needed)
  - **Memory sharing** across sequences (e.g., shared prompt prefixes in parallel decoding, beam search reuse blocks)
  - Enables higher batch sizes → higher throughput
- **This is the core win behind vLLM's 10-20x serving speedup**

<img width="1100" height="605" alt="image" src="https://gist.github.com/user-attachments/assets/e165c46d-c595-4e30-b2f1-5f7025dc4c91" />


### GQA (Grouped Query Attention) & Multi-Query Attention (MQA)
- **Problem**: Multi-head attention has (num_heads × head_dim) K/V tensors per layer — KV cache is huge
- **MQA (Multi-Query Attention)**: all query heads share **one** K/V pair — drastically reduces KV cache size (e.g., from 32 heads worth to 1)
  - Tradeoff: sometimes slight accuracy drop vs full MHA
  - Used in: fast models (Falcon, T5-large)
- **GQA (Grouped Query Attention)**: middle ground — query heads grouped, each group shares one K/V pair (e.g., 8 query head groups share 8 K/V pairs, vs 32 full pairs)
  - Better accuracy than MQA while still reducing cache size
  - Increasingly common in new models (Llama 3.1, Mistral)
- **Interview angle**: GQA is an easy win for inference efficiency, worth mentioning if asked about serving optimizations

<img width="800" height="443" alt="image" src="https://github.com/user-attachments/assets/73de4166-049c-4915-98a7-0eac56fa8585" />


### Other Serving Optimizations
- **Continuous batching**: dynamically add/remove requests from batch mid-generation instead of static batching (reduces GPU idle, enables higher throughput)
- **Speculative decoding**: small "draft" model proposes multiple tokens, large model verifies in parallel → speeds up generation
- **FlashAttention**: fuses attention computation to reduce memory I/O (reads/writes to HBM), not an approximation — exact attention, just faster via tiling
- **Flash-Decoding**: variant of FlashAttention optimized for single-token decode phase (different compute/memory tradeoff than prefill)


### Temperature & Sampling (Inference Parameters)
During inference, LLMs don't just pick the highest-probability next token; they use **sampling** to introduce variability.

| Parameter | Effect | Typical Range |
|---|---|---|
| **Temperature** | Controls randomness. `logits / temp` before softmax. Low=deterministic, high=creative | 0.1-2.0; default 1.0 |
| **Top-K** | Sample from top K most likely tokens (ignores tail of distribution) | 20-100 |
| **Top-P (Nucleus)** | Sample from smallest set of tokens with cumulative probability > P | 0.8-0.95 |
| **Min-P** | Only sample tokens with prob ≥ min_p × max_prob; more adaptive than fixed Top-K | 0.05-0.2 |

**Interview framing:** "Temperature rescales logits before softmax — low temp sharpens distribution (deterministic), high temp flattens it (more random). Top-P/Top-K directly prune the tail of the distribution to avoid sampling absurdities."

**Common gotcha:** Temperature 0 = argmax (always pick highest prob), NOT "no randomness" — some implementations special-case this.

---

## 7. LLM Training Pipeline: Pretraining → SFT → RLHF/DPO

*This is "how does a base model become ChatGPT" — commonly asked to distinguish candidates who've only ever called an API from those who understand what's under the hood.*

### Stage 1: Pretraining
- Train on massive unlabeled text corpora with a simple objective: **next-token prediction** (causal language modeling)
- This is where the model learns grammar, facts, reasoning patterns, and world knowledge — purely from predicting the next token at scale
- Result: a **base model** — fluent but not necessarily helpful, obedient, or safe. It completes text; it doesn't "follow instructions" reliably (e.g., asked a question, it might continue the question rather than answer it)
- Extremely expensive (thousands of GPUs, weeks/months) — this is the stage almost nobody outside a few labs actually does
- **Data quality matters enormously** — filtering out low-quality text, removing PII, deduplication all improve downstream performance

### Stage 2: Supervised Fine-Tuning (SFT)
- Fine-tune the base model on a curated dataset of **(prompt, ideal response)** pairs, written/curated by humans
- Teaches the model the "assistant" format/behavior: follow instructions, answer directly, adopt a helpful tone
- Still just standard supervised learning (cross-entropy loss on the target response tokens) — nothing exotic yet
- This is the stage most "instruction-tuned" or "chat" open-source models stop at (e.g., early Alpaca-style models)
- **Data scale**: typically 5K-100K high-quality examples (vs billions for pretraining)

### Stage 3: Alignment — RLHF (Reinforcement Learning from Human Feedback)
1. **Collect preference data**: humans rank multiple model outputs for the same prompt (A better than B, or A>B>C ranking)
2. **Train a Reward Model (RM)**: a separate model learns to predict human preference scores from this ranked data (typically via Bradley-Terry or similar ranking loss)
3. **RL fine-tuning (typically PPO — Proximal Policy Optimization)**: 
   - The SFT model is the "policy" to optimize
   - The RM outputs a scalar reward for each generation
   - Use PPO to maximize reward while staying close to the SFT model (KL-divergence penalty prevents drift / "reward hacking")
- **Goal**: align the model with human preferences (helpfulness, harmlessness, honesty) beyond what SFT alone captures
- **Downside**: RLHF/PPO is complex, unstable to train, and requires maintaining 4 models in memory simultaneously (policy, reference/SFT copy, reward model, value model) — expensive and finicky

### DPO (Direct Preference Optimization) — the modern simplification
- **Key insight**: you can mathematically derive a loss function that **directly optimizes on preference pairs (chosen vs rejected response)** without needing to train a separate reward model or run RL at all
- **How it works**: DPO loss directly increases the likelihood of the "chosen" response relative to the "rejected" one, using the SFT model as an implicit reference
- **Benefits**: 
  - Much simpler to implement, more stable
  - No separate RM training
  - No RL loop (no PPO complexity)
  - Cheaper — single forward/backward pass per pair
  - This is why DPO (and variants like KTO, ORPO, IPO) have largely replaced full RLHF/PPO pipelines in practice
- **Interview soundbite:** *"DPO reformulates RLHF's reward-modeling + RL problem as a single classification-style loss on preference pairs — same end goal (align to human preference), far simpler optimization."*

### GRPO (Group Relative Policy Optimization) — emerging alternative

- **Key insight** (vs DPO): instead of optimizing on preference pairs (chosen vs rejected), GRPO groups outputs and optimizes **relative rankings within groups**
- **How it works**: 
  - Generate multiple responses for the same prompt (e.g., 4 variants)
  - Rank them (or get human rankings)
  - Optimize so higher-ranked responses get higher likelihood, lower-ranked get penalized
  - No separate reference model or implicit reward — uses group-relative ranking directly
- **Advantages over DPO**:
  - More sample-efficient (one ranking covers multiple comparisons)
  - Handles ties/indifference naturally (same rank = no penalty)
  - Better scaling with batch of generations
  - More flexible ranking (can use soft rankings, not just binary preference)
- **Used by**: DeepSeek and other labs optimizing for reasoning models
- **When it wins**: when you have multiple outputs per prompt (e.g., diversity sampling, generation variants) and want to rank them holistically
- **Tradeoff vs DPO**: slightly more complex to implement, requires multiple outputs per prompt (vs simple pairs), but potentially better convergence

**Quick comparison of DPO variants:**

| Variant | Input | Optimization | Sample Efficiency | Complexity |
|---|---|---|---|---|
| **DPO** | Preference pairs (better vs worse) | Binary ranking loss | Baseline | Simple |
| **GRPO** | Multiple outputs + ranking | Group-relative ranking loss | Higher (multiple comparisons per prompt) | Moderate |
| **IPO** | Preference pairs | Implicit ranking with different loss | Similar to DPO | Simple |
| **ORPO** | Prompt + preferred + dispreferred | Odds ratio + reference | Baseline + adds regularization | Moderate |
| **KTO** | Binary feedback (good/bad, not pairs) | Class-relative optimization | Can use binary labels | Simple |



**Interview framing:** "GRPO handles group rankings better than pairwise comparison — useful when you're sampling multiple outputs and want holistic ranking. DPO is simpler if you already have preference pairs."

### Quick Comparison
| Stage | Objective | Data Needed | Output | Cost |
|---|---|---|---|---|
| Pretraining | Next-token prediction | Massive unlabeled text (TB scale) | Base model | Massive (1000s GPUs, weeks+) |
| SFT | Supervised loss on ideal responses | (prompt, response) pairs (10K-100K) | Instruction-following model | Moderate |
| RLHF (PPO) | Maximize learned reward w/ KL penalty | Human preference rankings + RM train | Aligned/"chat" model | High (4 models in memory) |
| DPO | Direct loss on preference pairs | Human preference rankings (chosen vs rejected) | Aligned/"chat" model | Low-moderate |
| GRPO | Direct loss on group rankings | Multiple outputs + human rankings per prompt | Aligned/"chat" model | Low-moderate |

<img width="800" height="611" alt="Untitled design" src="https://gist.github.com/user-attachments/assets/2f99dc3e-3749-4df0-b52b-93aa909416c4" />

---

## 8. Prompt Engineering & Few-Shot Learning

### Core Techniques
| Technique | Description |
|---|---|
| **Zero-shot** | No examples given, just instructions |
| **Few-shot** | Provide 2-5 examples in the prompt to steer format/style |
| **Chain-of-Thought (CoT)** | "Let's think step by step" — asking the model to reason before answering, improves accuracy on multi-step/math/logic tasks |
| **Self-consistency** | Sample multiple CoT reasoning paths, take majority-vote answer — more robust than single CoT |
| **ReAct** | Interleaves **Reasoning** + **Acting** (tool calls) — "Thought → Action → Observation" loop, foundation of most agent frameworks |
| **Structured Output / JSON mode** | Constrain output to a schema (via function calling or grammar-constrained decoding) — critical for pipelines that parse LLM output programmatically |

### System vs User vs Assistant Prompts
- **System prompt**: sets persona, constraints, high-level behavior (persists across turns)
- **User prompt**: the actual query/instruction
- **Assistant prompt (prefill)**: can be used to steer output format by starting the assistant's response for it (supported by some APIs)

### Function / Tool Calling
- Model outputs a structured request (function name + args) instead of free text when it decides a tool is needed
- Workflow: define tool schemas → model chooses tool + arguments → application executes → result fed back to model → model produces final answer
- Common failure mode: model hallucinates arguments or calls a tool when it doesn't need to — mitigated by clear tool descriptions + examples

### Prompt Injection (security angle — often asked)
- **Direct injection**: user directly tries to override system instructions ("ignore previous instructions...")
- **Indirect injection**: malicious instructions hidden inside retrieved documents/web content that the LLM ingests as "data" but treats as instructions
- Mitigations: clearly delimit untrusted content, input/output guardrails, least-privilege tool access, never let retrieved content directly trigger actions without validation

### Practical Prompting Tips (best practices)
- Be explicit about output format (ask for XML/JSON tags, not just "please format nicely")
- Give positive AND negative examples ("do this, not that")
- Put critical instructions at the start and end of long prompts (mitigates "lost in the middle")
- Use delimiters (```, XML tags) to clearly separate instructions from data
- **Lost in the middle**: LLMs attend less to middle-of-context info — put critical retrieval results at the beginning or end of context

---

## 9. Agents

### What Makes an "Agent"
An LLM wrapped in a loop that can: **reason → decide on an action (tool call) → observe result → repeat** until it reaches a final answer or goal — as opposed to a single-shot prompt→response.

### ReAct Pattern (core agent loop)
```
Thought: I need to find the current weather to answer this
Action: call_weather_api(city="London")
Observation: 15°C, cloudy
Thought: I now have enough info
Final Answer: It's 15°C and cloudy in London
```

### Planning vs Execution Separation
- **Planner**: breaks a complex goal into a sequence of sub-tasks (can be a separate LLM call or the same model in a "plan" step)
- **Executor**: carries out each sub-task, possibly calling tools
- Separating these improves reliability on complex multi-step tasks vs asking one model to plan-and-act simultaneously

### Multi-Agent Orchestration
- **Supervisor/worker pattern**: one orchestrator LLM delegates sub-tasks to specialized worker agents (e.g., a "researcher" agent, a "coder" agent), then synthesizes results
- **Debate/critique pattern**: one agent generates, another critiques/verifies — improves output quality (self-refine loops)
- Frameworks to know by name: LangGraph, AutoGen, CrewAI (implementation detail, not usually deep-dived, but good to mention)

### Common Agent Failure Modes (interviewers love asking about these)
- **Infinite loops**: agent keeps calling the same tool without making progress → mitigate with max iteration limits, loop detection
- **Tool misuse**: wrong tool chosen, or valid tool called with malformed/hallucinated arguments → mitigate with strict schemas + validation
- **Compounding errors**: small mistake early in a multi-step chain propagates and amplifies → mitigate with intermediate verification/checkpoints
- **Context window bloat**: long tool-call histories eat up context → mitigate with summarizing/pruning intermediate steps
- **Lack of grounding**: agent "hallucinates" that an action succeeded — always verify via actual observation, not assumption

### Agents vs Simple RAG (a common distinguishing question)
- RAG: single retrieval → single generation (no decision-making loop)
- Agent: can decide **whether/when/how many times** to retrieve, use other tools, and iterate — more flexible but less predictable and more expensive/slower

---

## 10. Model Context Protocol (MCP)

*Emerging standard for connecting LLMs to tools/data sources; increasingly asked in interviews, especially at AI-native companies.*

### What is MCP?

**Core idea:** A standardized, transport-agnostic protocol for connecting Claude (or other LLMs) to external tools, data sources, and services — moving away from ad-hoc tool definitions toward a reusable, composable standard.

**Why it matters:**
- Decouples tool implementation from LLM application code
- Enables code reuse across projects/teams
- Makes tool schemas and behaviors explicit + versioned
- Simplifies building complex agent workflows

**Unlike basic function calling:**
- Function calling: you manually define tool schemas, model calls them, you execute — tightly coupled
- MCP: standardized protocol → tool server exposes capabilities, Claude client discovers and uses them — loosely coupled

### MCP Architecture (Simple)

```
Claude App (Client)
    ↕ (MCP Protocol over stdio/HTTP/WebSocket)
    MCP Server
        ├─ Tool definitions (schemas, descriptions)
        ├─ Resource handlers (files, APIs, DBs)
        └─ Callback execution logic
```

**Flow:**
1. Client (Claude app) initiates connection to MCP server
2. Server advertises available tools + resources + prompts (discovery)
3. Client (model) decides to use a tool → sends request over protocol
4. Server executes → returns result → client feeds back to model
5. Model generates final response

### Key MCP Concepts

| Concept | Purpose |
|---|---|
| **Tools** | Functions the model can call — analogous to function calling, but standardized schema + versioning |
| **Resources** | Stateful data sources (files, API endpoints, DB connections) that tools can access |
| **Prompts** | Reusable prompt templates with context — versioned, shareable across teams |
| **Sampling** | Model can delegate to the server (e.g., "sample 10 variations of this query") |

**Interview framing:** "MCP is to tools what Docker is to deployments — standardization + portability + composition."

### MCP Servers (Examples)

**Built-in/Popular MCP servers:**
- **filesystem**: read/write files, directory listing
- **postgres**: SQL query tool for PostgreSQL databases
- **memory**: persistent key-value storage for multi-turn context
- **web-search**: integration with web search APIs
- **git**: interact with Git repositories
- **slack**: send messages, retrieve conversation history
- **github**: create issues, read repos, trigger workflows

**Why this matters:** instead of building search → retrieval → LLM each time, you just plug in the MCP server that wraps it. Team A's search implementation becomes reusable for Team B.

### Client vs Server Model

**MCP Server** (tool provider):
- Exposes capabilities (tools, resources)
- Handles requests from clients
- Manages authentication/rate limiting
- Examples: custom API wrapper, database connector, file system service

**MCP Client** (Claude app):
- Connects to one or more servers
- Sends tool-call requests
- Handles execution results
- Passes results back to LLM
- Examples: Claude app, agentic framework, standalone tool

**Key insight:** a process can be both (servers can compose other servers) — enables complex multi-service architectures.

### Transport Flexibility

MCP doesn't mandate transport — supports:
- **stdio**: subprocess communication (simplest, local)
- **HTTP**: REST over network (common for remote services)
- **WebSocket**: bidirectional, real-time (good for streaming/multi-turn)
- **SSE (Server-Sent Events)**: push notifications from server to client

**Interview angle:** "This flexibility means you can start with a local subprocess during dev, scale to HTTP in prod, add WebSocket for real-time without changing tool code."

### MCP vs Function Calling vs Agent Frameworks

| Aspect | Function Calling | MCP | Agent Framework (LangGraph) |
|---|---|---|---|
| **Tool definition** | Inline, per app | Standardized, versioned | Framework-specific |
| **Reusability** | Low (tightly coupled) | High (composable) | Medium (some portability) |
| **Complexity** | Simple, single request/response | Moderate, protocol overhead | High, full orchestration |
| **Scalability** | Single model calls tools | Multiple models + servers | Complex multi-agent workflows |
| **Use case** | Quick prototypes, simple queries | Production systems, team reuse | Complex reasoning, long-horizon tasks |

**Interview soundbite:** "MCP is for when you need tools to be reusable, versioned, and independent of the LLM application code. Function calling is simpler if you don't have those requirements."

### Real-World MCP Pattern

**Scenario:** You have a RAG system + a database query tool + a web search fallback.

**Without MCP:**
```python
# In your main app
def handle_query(q):
    # Manually implement each tool
    if should_search(q):
        results = web_search(q)
    elif should_query_db(q):
        results = db_query(q)
    else:
        results = rag_retrieval(q)
    return llm.complete(prompt + results)
```

**With MCP:**
```python
# Each tool is an independent MCP server
# Your app just connects and lets the model decide
client = MCPClient([
    "web_search_server",
    "database_server", 
    "rag_server"
])
response = client.chat("what's the Q3 revenue?")
# Model calls the right tool automatically
```

Benefits:
- Database team can update schema without touching app code
- Web search service can be swapped/updated independently
- Same tools work across multiple Claude apps

### When to Use MCP (and When Not To)

**Use MCP:**
- Multiple apps need the same tools
- Tools will evolve/be maintained separately
- Team has different tool owners (infrastructure, data, ML)
- Long-term production system where tool reuse matters
- Building modular agent architectures

**Don't use MCP (use function calling instead):**
- Quick one-off prototype
- Single tool, single app
- Tool logic is simple and unlikely to change
- Low latency critical (MCP protocol overhead)

### MCP in System Design Interviews

**If asked about building a complex agent system:**
- Mention MCP for tool modularity
- "We'd have an MCP server for each critical service (search, database, knowledge graph) so teams can own + update independently"
- Separate tool evolution from app logic
- Easier to test (mock MCP servers in tests)

**If asked about scaling tool infrastructure:**
- MCP servers can be deployed independently (containerized, versioned)
- Each server has its own auth, rate limiting, monitoring
- Client app just defines which servers to connect to

### Interview Soundbites on MCP

- *"MCP is a standardized protocol for connecting LLMs to tools — moves tool definitions from inline code to first-class services"*
- *"Unlike function calling where tool logic is embedded in your app, MCP servers are independent, versioned, and composable"*
- *"When multiple teams need the same tools (search, database, APIs), MCP makes them reusable and independently updatable"*
- *"MCP clients can connect to multiple servers simultaneously — the model decides which tool to use, not your orchestration logic"*
- *"Transport is flexible (stdio, HTTP, WebSocket) — start simple locally, scale to distributed microservices without changing tool code"*

---

## 11. Embeddings Deep-Dive

*RAG and semantic search both live and die by embedding quality — worth understanding beyond "it turns text into vectors."*

### What an Embedding Actually Is
- A dense vector representation of text (or image/audio) such that **semantic similarity ≈ geometric proximity** in vector space
- Learned via contrastive objectives: pull semantically similar pairs closer, push dissimilar pairs apart

### Similarity Metrics — know when each applies
| Metric | Formula intuition | When to use |
|---|---|---|
| **Cosine Similarity** | Angle between vectors, ignores magnitude | Most common for text embeddings — magnitude often just reflects text length, not meaning |
| **Dot Product** | Cosine × magnitude of both vectors | Faster to compute (no normalization step); equivalent to cosine if vectors are pre-normalized to unit length |
| **Euclidean (L2) Distance** | Straight-line distance | Sensitive to magnitude; common in non-text embedding spaces (e.g., some image embeddings) |

**Interview trap:** if embeddings are already normalized to unit length, cosine similarity and dot product give the **same ranking** — many vector DBs default to dot product for this reason (cheaper, no division).

### How Embedding Models Are Trained (contrastive learning basics)
- **Contrastive loss** (e.g., InfoNCE): given an anchor and a positive (similar) example, push their embeddings together while pushing apart embeddings of negatives (other unrelated examples in the batch)
- **In-batch negatives**: cheap trick — treat every other example in the training batch as a "negative" for the current anchor, no need for explicitly labeled negatives
- **Hard negative mining**: deliberately include negatives that are *superficially* similar but semantically different (harder to distinguish) — significantly improves retrieval quality over random negatives
- Sentence-embedding-specific approaches: **Siamese/bi-encoder** architecture (two texts encoded independently, compared via similarity) vs **cross-encoder** (both texts fed jointly into one model — much more accurate but can't be pre-computed/indexed, hence used only for reranking a small candidate set, not full-corpus search)

### Bi-encoder vs Cross-encoder (a very common question)
| | Bi-encoder | Cross-encoder |
|---|---|---|
| Architecture | Encodes query & doc **separately** | Encodes query+doc **together** (joint attention) |
| Speed | Fast — docs pre-embedded & indexed once | Slow — must run inference per query-doc pair |
| Accuracy | Lower (no cross-attention between query/doc) | Higher (full interaction between query & doc) |
| Use case | Initial retrieval over millions of docs | Reranking a small top-k candidate set |

<img width="800" height="523" alt="image" src="https://gist.github.com/user-attachments/assets/350ef87e-7ccd-4156-8a8c-6e4fd064b98d" />


### Fine-tuning Your Own Embedding Model (when off-the-shelf isn't enough)
- Useful when domain vocabulary is highly specialized (legal, medical, internal jargon) and general embeddings cluster things incorrectly
- Typically done via contrastive fine-tuning on domain-specific (query, relevant-doc) pairs, often starting from a strong open-source base (e.g., `bge`, `e5`) rather than training from scratch
- Evaluate with retrieval metrics (Recall@k, MRR) on a held-out labeled set before/after fine-tuning to confirm actual improvement

### Dimensionality Tradeoffs
- Higher dimension → captures more nuance, generally better recall, but more storage + slower search
- **Matryoshka embeddings** (newer technique, e.g. OpenAI's `text-embedding-3`): trained so that truncating the vector (e.g., using only the first 256 of 1536 dims) still gives a usable, well-ordered embedding — lets you trade off storage/speed vs accuracy *after* embedding, without re-embedding everything

---

## 12. RAG (Retrieval-Augmented Generation)

### RAG Pipeline — Step by Step

1. **Document Ingestion**: load raw docs (PDF, HTML, DB records)
2. **Chunking**: split into smaller pieces
   - Fixed-size chunking (with overlap, e.g., 512 tokens, 50 overlap)
   - Semantic chunking (split on meaning/topic shifts)
   - Recursive chunking (paragraph → sentence fallback)
   - **Tradeoff**: small chunks = precise retrieval but less context; large chunks = more context but noisier/less precise
3. **Embedding**: convert chunks → dense vectors (e.g., `text-embedding-3`, `bge-large`, `e5-mistral`)
4. **Indexing/Storage**: store vectors in a vector DB (Pinecone, Weaviate, Qdrant, FAISS, Milvus, pgvector)
   - Index types: **HNSW** (Hierarchical Navigable Small World — fast approximate nearest neighbor, most common), **IVF**, flat (exact but slow)
5. **Query Embedding**: embed the user query with the same embedding model
6. **Retrieval**: similarity search (cosine similarity / dot product) → top-k chunks
   - Often combined with **hybrid search**: dense (semantic) + sparse (BM25/keyword) retrieval, merged via score fusion (e.g., RRF - Reciprocal Rank Fusion)
7. **Reranking**: a cross-encoder reranker (e.g., Cohere rerank, `bge-reranker`) re-scores top-k candidates for relevance — more accurate than embedding similarity alone but slower (used only on a small candidate set)
8. **Context Construction**: assemble retrieved chunks into a prompt (with citations/metadata)
9. **Generation**: LLM generates answer conditioned on retrieved context
10. **(Optional) Post-processing**: citation attribution, hallucination check, guardrails

### Model Selection Process (for RAG components)

**Embedding model selection:**
- Check **MTEB leaderboard** benchmarks for retrieval task performance
- Consider: dimension size (higher = better recall but more storage/compute), max token length, domain match (general vs code vs multilingual), latency
- Common choices: OpenAI `text-embedding-3-small/large`, `bge-large-en`, `e5-mistral-7b`, Cohere embed-v3

**Generator LLM selection:**
- Context window needed (how much retrieved content + query fits)
- Latency/cost requirements (larger flagship models vs smaller/faster tiers)
- Instruction-following quality for grounded/citation-based answers
- Open-source vs closed API (data privacy, cost at scale, self-hosting complexity)

**Vector DB selection:**
- Scale (millions vs billions of vectors), filtering needs (metadata filters), managed vs self-hosted, latency SLAs

**Evaluation-driven selection**: always A/B test candidate models against a labeled eval set using retrieval metrics (below) — don't pick based on marketing benchmarks alone.

### RAG Optimizations

#### **Semantic Caching**
- Cache query→response (or query→retrieved-chunks) pairs
- On a new query, check semantic similarity (via embedding) to cached queries — if similarity > threshold, return cached result instead of re-running full pipeline
- Reduces latency + cost for repeated/similar queries (e.g., FAQ-style traffic)
- Risk: stale answers if underlying data changes — needs cache invalidation/TTL strategy

#### **Model Routing**
- Use a small/cheap classifier (or LLM) to route queries to different models based on complexity
- E.g., simple factual query → small fast model; complex reasoning → larger model
- Can also route between **RAG vs no-RAG** (e.g., "hi, how are you" doesn't need retrieval)
- Reduces cost and latency at scale

#### **Retrieving Fewer, Higher-Quality Chunks**
- More chunks ≠ better — increases prompt length (cost, latency) and can introduce **"lost in the middle"** problem (LLMs attend less to middle-of-context info)
- Optimization strategies:
  - **Reranking** to surface only the most relevant top-k (e.g., retrieve 50 → rerank → keep top 3-5)
  - **Contextual compression**: summarize/trim retrieved chunks to only relevant sentences before passing to LLM
  - **Smaller, well-bounded chunks** with good metadata so retrieval precision is higher, needing fewer chunks
  - **Query rewriting/decomposition**: rewrite vague queries into more specific sub-queries for more targeted retrieval

#### Other RAG optimizations worth knowing
- **HyDE (Hypothetical Document Embeddings)**: generate a hypothetical answer first, embed that, use it to retrieve (often better than embedding the raw query)
- **Query expansion**: generate multiple query variants to improve recall
- **Multi-vector retrieval / parent-child chunking**: retrieve small precise chunks but return the larger parent chunk for context
- **Metadata filtering**: pre-filter by date/source/category before vector search to narrow search space

### Production Latency Optimization (End-to-End)

*Beyond retrieval-quality tricks above, a working RAG system also has to hit a latency budget. This is the systematic, production-engineering side of RAG optimization — the interview answer that separates "knows RAG concepts" from "has actually run one in production."*

**Where latency actually comes from:** a naive mental model is `total latency ≈ request overhead + retrieval + reranking + prompt/prefill + generation`, but many stages can run in parallel, so profiling — not intuition — must drive optimization. In a real trace, **LLM generation (TTFT + token generation) is almost always the dominant cost**, often 50%+ of total latency, with reranking a distant second and vector search/embedding a small fraction. This is why shaving milliseconds off vector search is rarely the highest-leverage fix.

**Define a latency budget (SLO) before optimizing anything** — e.g. p50 < 1.5s, p95 < 3s, p99 < 5s — and break it into a per-stage budget (embedding, retrieval, reranking, prompt build, TTFT, generation). Always track **p50/p95/p99**, not just the average: an average can look fine while a meaningful tail of users has a terrible experience (95 users at 1s + 5 users at 10s still "looks okay" on average).

**TTFT vs total latency are different problems.** Time-To-First-Token (TTFT) is driven by prompt length/prefill; total generation time is driven by output length/decode speed — measure them separately. **Streaming** (returning tokens as they're generated instead of waiting for the full response) is the single biggest *perceived*-latency win, but it does not reduce actual compute time — it only improves time-to-useful-output. This distinction is frequently the crux of a "how would you make RAG feel faster" interview question.

**Caching, at multiple layers (cheapest wins, apply in this order):**

| Layer | What's cached | Risk |
|---|---|---|
| Query embedding cache | Hash(query) → embedding vector, skips re-embedding repeated queries | Low risk, easy win |
| Semantic query cache | Similar (not identical) queries reuse cached embedding/results via a similarity threshold | Threshold too loose → wrong reuse; must be tuned empirically |
| Retrieved-results cache | Query → top-k doc IDs, skips the vector DB round-trip | Staleness if the knowledge base changes — key must include KB/embedding-model version |
| Full LLM answer cache | Question → final answer, skips the whole pipeline | Dangerous for dynamic/time-sensitive info; needs explicit invalidation when source docs change |
| Prompt/prefix (KV) caching | Reused system-prompt/instruction prefixes reuse cached KV state at the inference layer | Infra-level; effective for repeated system prompts, less so for highly variable prefixes |

**Retrieval-side latency levers:**
- **ANN over exact search**: HNSW/IVF/PQ trade a small amount of recall for large speedups over brute-force nearest neighbor at scale (millions of vectors) — tune ANN parameters against a recall/latency benchmark, don't just pick the fastest setting.
- **Metadata filtering before vector search**: narrowing 10M vectors down to a tenant/department-scoped subset before the ANN search improves both relevance and latency.
- **Hybrid retrieval (dense + BM25) run in parallel, not sequentially**: `max(dense, bm25)` instead of `dense + bm25` — a straightforward async/concurrency win that's easy to miss if retrieval calls are written sequentially.
- **Query rewriting/multi-query fusion adds real latency** (an extra LLM call before retrieval even starts) — only justified if it measurably improves *end-to-end* answer quality, not just an offline retrieval metric.

**Reranking discipline:** rerankers (cross-encoders) are accurate but slow — the common mistake is reranking too many candidates (e.g., reranking all 1000 retrieved instead of the top 50). Use a **confidence-gated reranker**: skip reranking when first-stage retrieval scores are already high-confidence, only invoke the reranker on ambiguous cases. Right-size retrieve→rerank→keep counts (e.g., 50→20→5) via evaluation, not by default.

**Context and prompt size control:** the goal isn't "give the LLM as much context as possible," it's "give it the smallest amount of high-quality context needed" — more chunks means longer prefill (higher TTFT, higher cost) and a worse lost-in-the-middle risk. This extends to **conversation history**: sending the full multi-turn history on every request grows unboundedly; summarize/compress older turns and keep only recent turns + relevant memory.

**Model-side levers:** **model routing** (small/fast model for simple queries, large model reserved for complex ones) is usually the biggest cost/latency lever after eliminating unnecessary pipeline stages. **Quantization** and **speculative decoding** matter for self-hosted serving but are advanced/late-stage optimizations — apply after the simpler wins (model choice, prompt size, caching, serving config) are exhausted.

**Infra and reliability:** colocate latency-sensitive services in the same region, use connection pooling/persistent connections/async I/O to avoid sequential network round-trips, and pre-warm models/connections to avoid cold-start latency spikes. Use bounded retries with timeouts and circuit breakers — an unbounded retry against an unhealthy dependency can silently consume the entire latency budget. Design graceful degradation (reranker times out → fall back to raw retrieval order; primary model unavailable → fall back to a smaller model) rather than letting one dependency fail the whole request.

**The optimization hierarchy (apply in this order, cheapest/highest-leverage first):**
1. **Eliminate work** — cache hits, skip RAG entirely for simple/greeting queries, skip reranking when unnecessary
2. **Parallelize work** — run dense + BM25 + metadata lookups concurrently instead of sequentially
3. **Reduce work** — smaller context, smaller top-k, shorter prompts, trimmed history
4. **Make the remaining work faster** — ANN, GPU serving, quantization, continuous batching
5. **Improve perceived latency** — streaming, progressive UI feedback

**Observability is what makes any of this possible.** Instrument every stage of the pipeline individually (embedding, dense search, BM25, fusion, reranking, document fetch, prompt build, TTFT, generation) so that "RAG is slow" becomes "reranking is 40% of p95 latency" instead of a guessing exercise. Without per-stage tracing, engineers tend to optimize the wrong thing (e.g., the vector DB) while the LLM generation call — which usually dominates — goes unexamined.

**Interview soundbite:** *"I wouldn't start by asking which vector database is fastest — I'd instrument the full pipeline, establish a p95/p99 latency budget, and find the actual bottleneck first. In most RAG systems that's LLM generation, not retrieval, so the highest-leverage fixes are usually eliminating unnecessary pipeline stages, caching, and streaming — not micro-optimizing vector search."*

**Common mistakes to flag proactively:** optimizing before profiling (assuming vector search is the bottleneck without measuring); increasing top-k to chase recall without accounting for the added reranking/prompt cost; sending the entire conversation history and full retrieved context to the LLM by default; assuming streaming reduces total compute time (it only improves perceived latency); and optimizing only p50 while p99 users silently have a much worse experience.

### Fallback Behaviors (Production Reality)
When retrieval fails or returns low-confidence results:

| Scenario | Fallback Strategy |
|---|---|
| **No relevant chunks retrieved** | Return honest "I don't know" instead of hallucinating; optionally escalate to human |
| **Low confidence score** | Use confidence threshold; if below, route to a more powerful model or human |
| **Outdated retrieved content** | Combine with real-time search (Google, internal APIs) for fresh info |
| **Contradictory sources** | Surface the conflict to user, cite all sources, let them decide; don't hallucinate consensus |
| **Ambiguous query** | Ask clarifying question instead of guessing; improves UX vs. wrong answer |

**Interview angle:** "A good production system knows when it doesn't know — fallback handling is as important as the happy path."

---

## 13. Multimodal Models (Vision + Language)

*Increasingly asked, and easier than deep vision knowledge.*

### Vision Transformers (ViT)
- **Idea**: treat image as a sequence of patches (e.g., 16×16 pixel patches), linearize → apply standard Transformer
- **Process**: image → patches → linear projection → position embeddings → standard transformer encoder
- **Advantage**: same architecture as language transformers, parallelizable, scales with image resolution
- **Tradeoff**: patch-based means fine details at very small scales can be lost; more patches = longer sequence = more compute

### CLIP (Contrastive Language-Image Pre-training)
- **Architecture**: dual-encoder — separate text encoder + image encoder, trained to align embeddings
- **Loss**: contrastive — pull (image, description) pairs together, push apart unrelated pairs
- **Why it matters**: learned a vision-language embedding space that's robust to paraphrasing, object variations, etc.
- **Use case**: zero-shot image classification, image-text retrieval, grounding text queries to images
- **Tradeoff**: dual-encoder is efficient for retrieval but weaker than models with cross-attention between modalities

### LLaVA-style Vision-Language Models
- **Architecture**: vision encoder (ViT or similar) → adapter/projection → language model
- **Process**: image → visual tokens → concatenate with text tokens → LLM processes unified sequence
- **Capabilities**: image understanding, OCR, visual reasoning, object counting, diagram interpretation
- **Limitation**: can hallucinate visual details or misinterpret complex scenes

### Multimodal Eval
- COCO Captions (caption quality), VQA (visual question answering accuracy), hallucination rate (false visual claims)
- Much harder to eval than text-only — often requires manual review or LLM-as-judge

---

## 14. Graph RAG & Knowledge Graphs

*Emerging area; shows up in interviews for advanced candidate pools.*

### Traditional RAG Limitations
- Text chunking loses relational structure (how entities connect)
- Dense retrieval finds similar text, not necessarily connected knowledge
- Hard to answer multi-hop questions (e.g., "What companies does the CEO of X founded earlier in their career?")

### Graph RAG Concept
- **Idea**: instead of retrieving text chunks, retrieve from a **knowledge graph** (entities + relationships as nodes/edges)
- **Build process**: extract entities & relationships from docs (via NER + relation extraction, often LLM-aided), build graph
- **Query process**: convert query to graph traversal, find relevant subgraph, pass to LLM as structured context
- **Benefit**: captures multi-hop reasoning, explicit entity relationships

### Knowledge Graphs (at a glance)
- Representation: nodes = entities, edges = relationships (e.g., "Obama" --[president_of]--> "USA")
- Construction: manual curation (expensive), automated extraction (error-prone but scalable), or hybrid
- Inference: entity linking, link prediction (what's the missing relationship?), path finding
- Storage: specialized graph DBs (Neo4j, ArangoDB) or generic triple stores (RDF, SPARQL)

### Graph RAG in Practice (Microsoft's approach)
- Chunk documents → extract entities/relationships → build hierarchical graph (local + global levels)
- Query: retrieve relevant subgraph, generate summaries at different levels, pass structured context to LLM
- Advantage: handles long-tail queries better than dense retrieval alone
- Tradeoff: much more complex to set up; extraction quality is bottleneck

**Interview angle**: "Graph RAG trades chunking simplicity for structured knowledge + multi-hop reasoning. Worth it when docs are dense with relationships (financial, legal, org structures)."

---

## 15. Distributed Training & Inference (surface-level, but expect it)

### Why It's Needed
Modern LLMs (billions of params) don't fit on a single GPU's memory, and training on huge datasets on one device would take forever — so both **model** and **data** get split across multiple GPUs/nodes.

### Parallelism Strategies
| Strategy | What's split | Notes |
|---|---|---|
| **Data Parallelism (DP)** | Same model copied on each GPU, different data batches | Gradients averaged (all-reduce) after each step; simplest, but model must fit on one GPU |
| **Tensor Parallelism (TP)** | Individual weight matrices split across GPUs (e.g., split a matmul across devices) | Needed when a single layer is too big for one GPU; high communication overhead, usually kept within a node (fast interconnect) |
| **Pipeline Parallelism (PP)** | Different **layers** of the model placed on different GPUs, data flows through like a pipeline | Reduces per-GPU memory, but introduces "bubble" idle time unless micro-batched carefully |
| **ZeRO / FSDP (Fully Sharded Data Parallel)** | Shards optimizer states, gradients, and/or parameters across GPUs (instead of full replication in DP) | Used by DeepSpeed/PyTorch FSDP — lets you train much larger models with the same GPU memory |

**Interview soundbite:** *"Data parallelism scales throughput, tensor/pipeline parallelism scale model size — real large-scale training combines all three (3D parallelism)."*

### Inference-side Scaling (ties back to Section 6)
- **Batching** (continuous batching) — throughput lever
- **Tensor parallelism at inference** — split a huge model across GPUs just to fit it in memory / lower latency
- **Quantization** — reduces per-GPU memory footprint, allows single-GPU serving
- **KV cache + PagedAttention** — the memory-efficiency lever specific to serving (see Section 6)

### Cost/Latency/Throughput Tradeoff (a favorite system-design thread)
- More GPUs / larger batch → higher throughput, but higher latency per request if batching too aggressively
- Time-to-first-token (TTFT) vs tokens/sec (generation speed) are usually optimized separately — TTFT dominated by prompt processing (prefill), generation speed dominated by decode step + KV cache reads

---

## 16. Observability & Evaluation

### Why It Matters
LLM/RAG systems are non-deterministic and can fail silently (hallucination, retrieval miss, prompt drift) — traditional software monitoring (uptime, latency) isn't enough.

### Observability (Tracing & Monitoring)

**What to trace in a RAG/LLM pipeline:**
- Input query, retrieved chunks (+ similarity scores), final prompt sent to LLM, model output, latency per stage, token usage/cost
- Tools: **LangSmith, Langfuse, Arize Phoenix, Weights & Biases (Weave), Helicone**

**Key production metrics:**
| Metric | Purpose |
|---|---|
| Latency (p50/p95/p99) | User experience, especially time-to-first-token |
| Token usage / cost per request | Budget tracking |
| Error/failure rate | API failures, timeouts, malformed outputs |
| Retrieval hit rate | % of queries where relevant doc was retrieved |
| User feedback (thumbs up/down) | Real-world signal, feeds back into eval sets |
| Drift detection | Are query patterns or input distributions changing over time? |

### Evaluation

#### LLM Output Evaluation
- **Reference-based**: BLEU/ROUGE (n-gram overlap — weak for open-ended generation), exact match (good for QA with short answers)
- **LLM-as-a-judge**: use a strong LLM to score outputs on rubrics (correctness, coherence, helpfulness) — scalable but has known biases (favors verbose answers, position bias in pairwise comparisons)
- **Human evaluation**: gold standard but expensive/slow — often used to validate LLM-judge alignment
- **Task-specific metrics**: for classification-style outputs, use precision/recall/F1; for structured output, JSON schema validity rate

#### RAG-specific Evaluation (RAGAS framework — know this cold)
| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the generated answer factually grounded in the retrieved context (no hallucination)? |
| **Answer Relevance** | Does the answer actually address the question asked? |
| **Context Precision** | Of the retrieved chunks, how many are actually relevant? |
| **Context Recall** | Did retrieval capture all the relevant info needed to answer? |
| **Context Relevance** | How relevant is retrieved context to the query overall? |

**Retrieval-only metrics:**
- **Precision@k / Recall@k**: relevance of top-k retrieved docs
- **MRR (Mean Reciprocal Rank)**: average of 1/rank of first relevant result
- **NDCG (Normalized Discounted Cumulative Gain)**: rewards relevant results appearing higher in ranking

#### Hallucination Detection
- Compare generated claims against retrieved context (entailment check, often via a smaller NLI model or LLM-judge)
- **Self-consistency check**: sample multiple generations, check agreement
- Citation-based grounding: force model to cite sources, verify citations map to real retrieved content

### Guardrails (often discussed alongside eval)
- Input guardrails: PII detection, prompt injection detection, off-topic filtering
- Output guardrails: toxicity filtering, schema validation, hallucination scoring before returning to user
- Tools: Guardrails AI, NeMo Guardrails, custom regex/classifier layers

---

## 17. Failure Diagnosis Playbooks

*The single most common category of open-ended AI/ML interview question is "X is broken/slow/wrong — walk me through how you'd find out why." This section is a repeatable framework, not a list of unrelated fixes: for every failure type below, triage into the right bucket first, then isolate, then fix. Interviewers are grading the triage step as much as the final answer.*

### The General Pattern
1. Split the symptom into mutually exclusive buckets before touching code — most wrong answers/slow requests/cost spikes have 3-5 plausible root causes that produce identical-looking symptoms.
2. Pick the cheapest diagnostic that discriminates between buckets (log a raw score, check a timestamp, count tokens) before reaching for a heavier fix.
3. Fix the actual bucket, don't shotgun-fix all of them — that's how you burn interview time, and in production, burn on-call time.

### 1. RAG Gives Wrong Answers

**First fork — is the answer grounded in what was retrieved, or not?**

| Bucket | How you tell | Likely cause | Fix |
|---|---|---|---|
| **Faithfulness failure** | Answer isn't supported by the retrieved chunks at all | Model ignoring context, falling back on parametric memory, or over-extrapolating | Strip the prompt down to bare context + question and retest; check if the hallucinated fact matches the base model's general knowledge (parametric leakage); lower temperature; add an explicit "only use the provided context" instruction |
| **Retrieval failure** | Answer faithfully reflects the retrieved chunks, but those chunks are wrong/irrelevant | Embedding model mismatch, bad chunking, similarity metric mismatch, metadata filter excluding relevant docs | Pull raw top-k + similarity scores and eyeball them; high score + wrong topic → embedding/domain issue; low scores across the board → genuine coverage gap in the index |
| **Source data failure** | Retrieved chunks are correct and relevant, answer faithfully reflects them, but they're simply wrong | Stale docs, conflicting versions, upstream data entry error | Audit the corpus, not the code — check doc freshness/versioning, look for contradictory sources on the same topic |
| **Silent pipeline bug** | Every stage looks fine in isolation, answer still wrong end-to-end | Index-time vs query-time embedding version drift, chunk-ID→text mapping scrambled on reindex, prompt template silently truncating context under a token budget | Check the boring stuff first: embedding model version pinned identically on both sides, chunk IDs resolve to the text you expect, log the *final assembled prompt* actually sent to the LLM — not what you think you sent |

**Interview soundbite:** *"A wrong RAG answer isn't one bug category — the first move is figuring out whether the failure is in retrieval, generation faithfulness, the source data, or a silent pipeline mismatch, because those four need completely different fixes."*

### 2. Latency Problems

| Symptom | Likely bucket | Isolate | Fix |
|---|---|---|---|
| High **TTFT** | Prefill-bound (long prompt) or queueing-bound (GPU saturation) | Check prompt token count vs typical; check queue depth/concurrent requests at the provider | Shrink context, prefix-cache the system prompt, reduce retrieved chunk count; or add capacity / provisioned throughput |
| Good TTFT, slow **tokens/sec** | Decode-bound | Check output length, KV cache size vs GPU memory bandwidth, batch size | Speculative decoding, smaller/faster model for this path, right-size batch |
| **P50 fine, P99 terrible** | Tail latency — one slow dependency blocking, GC pause, a hot/overloaded shard | Trace individual slow requests, not aggregates; check for retries without timeouts | Timeouts + circuit breakers on downstream calls, shard rebalancing, isolate noisy neighbors |
| **Latency degraded gradually over weeks**, no code change | Index growth past ANN sweet spot, traffic pattern shift, silent chunk-count creep | Compare index size/query volume now vs when it was fast; check if a routing/chunk-count parameter drifted | Reindex/reshard the vector DB, re-tune HNSW parameters, audit config drift |

**Interview soundbite:** *"In a RAG or agent pipeline, generation latency almost always dominates — so the highest-leverage fixes are semantic caching and streaming, not shaving milliseconds off vector search."* (See Section 12's "Production Latency Optimization" for the full end-to-end breakdown and optimization order.)

### 3. Cost Blowups

- **"Bill 3x'd, no code changed"** → check traffic growth first (boring, often the real answer), then a silent prompt-length regression (debug logging leaking into the prompt, a chunk-count bug retrieving 50 instead of 5), then a retry storm burning tokens on repeated failures.
- **"Cut inference cost 50% without hurting quality"** → model routing (cheap model for easy queries), prompt/prefix caching, trimming redundant context, distillation/fine-tune to a smaller model for the common case, batching where real-time isn't required.

### 4. Fine-Tuning Failures (beyond "loss fine, output garbage" in the coding/debugging section)

| Symptom | Likely cause | Fix |
|---|---|---|
| Model got *worse* at things it used to do well | Catastrophic forgetting — LR too high, fine-tune set too narrow | Lower LR, mix in general-purpose data (replay), use LoRA to limit drift |
| Training loss ↓ but eval score flat/worse | Overfitting to train distribution, eval/train leakage, loss doesn't correlate with the metric you actually care about | Check eval set independence, correlate loss vs target metric on a held-out slice before trusting either |
| Multi-GPU slower than single-GPU | Communication overhead dominating — wrong parallelism strategy for the hardware | Keep tensor parallelism within fast-interconnect nodes; check actual GPU utilization vs time spent in all-reduce |

### 5. Agent Failures (diagnosis, not just the failure-mode list in the Agents section)

- **Stuck in a tool-call loop**: instrument by logging (tool, args) tuples per step and flagging repeats — "add a max iteration limit" is the fix, but the interview wants to hear *how you'd detect it happening in production*, not just cap it blindly.
- **Agent claims an action succeeded but didn't**: don't trust the model's self-report — verify via the actual tool/API response (a write returned 200, a file exists, a record's `updated_at` changed) before letting the agent proceed.

### 6. Forecasting-Specific Failures (siblings to the backtest-vs-production drill in the coding section)

- **Consistently biased in one direction** (not just "inaccurate") → check for a systematic issue: promo effects not fully captured, or a log-transform back-transformation bug (a very common silent one — forgetting to exponentiate back after modeling in log space skews every prediction the same direction).
- **New-SKU forecasts are terrible** → check whether cold-start fallback logic (analogous product, hierarchical pooling) is actually firing, or whether the model is silently defaulting to zero/global average for unseen IDs.

### 7. "Eval Looks Great, Users Are Unhappy"

- Golden set doesn't represent real production traffic distribution (built from easy/clean examples, users ask messier things)
- LLM-as-judge bias inflating scores (verbosity bias, position bias) — validate against a human-labeled sample
- Eval measuring the wrong thing entirely (fluency/coherence instead of correctness/faithfulness)
- **A/B test shows no significant difference** — before concluding "no effect," check if the test was even powered to detect one at that sample size; a null result and an underpowered test look identical from the outside

**Interview soundbite:** *"A failure-diagnosis question is really testing whether you triage before you debug — wrong answers, slow requests, and cost spikes all have several plausible root causes that look identical from the outside, and picking the wrong bucket first wastes the whole investigation."*

---

## 18. Testing & Eval-Driven Development for LLM Apps

*The natural bridge between "Evaluation" (Section 16) and real engineering practice — how do you actually prevent regressions in a non-deterministic system?*

### The Core Problem
Traditional unit tests assert exact outputs. LLM outputs are non-deterministic and open-ended — you can't `assertEqual(response, "expected string")`. Testing strategy has to shift from exact-match to **quality-threshold** and **regression-detection** based approaches.

### Golden Datasets
- Curate a fixed set of representative (input, expected-behavior) examples — not necessarily exact expected strings, but pass/fail criteria (contains key facts, doesn't contain X, passes a rubric check)
- Should include: common cases, edge cases, known-tricky past failures (regression set), adversarial/red-team examples
- Grows over time — every production bug found should become a new golden test case (prevents the same failure recurring silently)

### Eval-Driven Development (EDD) Workflow
1. Define success criteria upfront (what does "good" look like — faithfulness? format compliance? latency budget?)
2. Build the golden/eval dataset **before** heavily iterating on prompts/pipeline
3. Every change (prompt tweak, model swap, chunk size change, new retrieval strategy) gets run against the full eval set
4. Compare aggregate scores before/after — ship only if it doesn't regress on the golden set, even if it "feels better" anecdotally
- This mirrors traditional CI/CD but with **eval scores** instead of pass/fail unit tests as the gate

### Prompt Regression Testing
- Treat prompts as versioned artifacts (like code) — track prompt versions alongside eval scores
- Automated regression suite: re-run golden dataset against every prompt/model change, flag score drops beyond a threshold
- Watch specifically for **silent regressions**: a prompt change that improves one category of queries but quietly degrades another — aggregate score alone can hide this, so segment eval results by query type/category

### A/B Testing & Online Evaluation
- Offline eval (golden set) catches known issues before deploy; **online A/B testing** (shadow traffic, canary rollout) catches issues only visible on real, messy production traffic
- Key online signals: user feedback (thumbs up/down), regeneration rate (user hit "retry" = implicit negative signal), session abandonment, escalation-to-human rate (for support bots)
- Feed production failures back into the golden dataset — the eval set should never be static

### Practical Tooling Mentions
- Frameworks: **RAGAS**, **DeepEval**, **Promptfoo** (prompt regression testing specifically), **LangSmith Evaluators**
- CI integration pattern: eval suite runs automatically on every PR that touches prompts/retrieval config, blocks merge if scores regress beyond threshold — same mental model as traditional test coverage gates

---

## 19. Coding / Whiteboard Practice

*Since you lean toward debugging/analyzing code — this is likely your highest-leverage section. Practice writing these from scratch (no framework), and practice explaining bugs out loud.*

### Implement from scratch

**1. Scaled dot-product attention (numpy)**
```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)  # causal mask
    weights = softmax(scores, axis=-1)
    return weights @ V
```
*Common follow-up: "why divide by √d_k?"* → keeps dot-product magnitude stable regardless of dimension, prevents softmax saturating into near one-hot (vanishing gradients).

**2. Cosine similarity search (brute-force, then explain how ANN/HNSW speeds it up)**
```python
import numpy as np

def cosine_sim(query, matrix):
    query = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query  # shape: (n_docs,)

def top_k(query, matrix, k=5):
    sims = cosine_sim(query, matrix)
    return np.argsort(-sims)[:k]
```
*Follow-up:* brute force is O(n·d) per query — explain why real systems use HNSW (graph-based approximate search, O(log n)) instead at scale.

**3. A minimal LoRA layer (PyTorch)**
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False  # freeze base weights

        in_dim, out_dim = base_layer.in_features, base_layer.out_features
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, r))  # zero-init B → starts as no-op
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```
*Follow-up:* "why zero-init B?" → so the LoRA delta is zero at the start of training (model behaves like the unmodified base model initially, stable training).

### Debugging exercises (talk through your reasoning out loud)

**"This RAG pipeline returns irrelevant chunks — debug it."**
Checklist to walk through:
1. Is the embedding model the same at index-time and query-time? (mismatch = broken)
2. Are chunks too large/small? (check chunk boundaries manually)
3. Is similarity metric consistent with how the vector DB was built (cosine vs dot vs L2)?
4. Is there a normalization mismatch (embeddings not normalized before cosine sim)?
5. Is metadata filtering silently excluding all relevant docs?
6. Print top-k raw scores — are they all low (retrieval genuinely has nothing relevant) or reasonably high but off-topic (embedding quality issue)?

**"Fine-tuned model's loss looks fine but outputs are garbage — debug it."**
1. Check tokenizer mismatch (using base tokenizer vs. one with different special tokens)
2. Check if a chat template / prompt format was applied consistently between training and inference
3. Check learning rate — too high with LoRA can destabilize despite low visible loss
4. Check if eval is being run in the right generation mode (e.g., missing EOS token handling → runaway generation)

**"LLM API calls are randomly slow — debug it."**
1. Check if it's TTFT (prompt/prefill heavy — long context?) vs total generation time (long output + no streaming?)
2. Check for retry storms / rate limiting causing backoff delays
3. Check batch size / concurrent request volume against provider's queuing behavior
4. Check if caching (semantic cache) is properly hitting for repeated queries

**"Forecast accuracy looks great in backtest but terrible in production — debug it."**
1. Check for **data leakage** in cross-validation (random k-fold instead of walk-forward — future info leaked into training)
2. Check if exogenous features used in backtest (e.g., promo flags) are actually **known in advance** at inference time, or if they were leaking future info during offline evaluation
3. Check for **concept drift** — has the underlying demand pattern shifted (new competitor, macro shock) since the model was trained?
4. Check if the backtest window covers the same seasonality/regime as the production period being evaluated

### Live "find the bug" drills (planted bugs — practice spotting these without running the code first)

These are the kind of snippets an interviewer hands you and says "this is broken, why?" Read each one, form a hypothesis before scrolling to the answer, then verify by running it mentally or on paper.

**Drill 1 — Attention mask applied in the wrong place**
```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    if mask is not None:
        weights = np.where(mask == 0, 0, weights)  # <-- bug
    return weights @ V
```
*Bug:* masking is applied **after** softmax by zeroing weights, instead of masking the raw `scores` with `-inf`/`-1e9` **before** softmax. Zeroing post-softmax weights doesn't renormalize the remaining probabilities — they no longer sum to 1, so the output is a silently-wrong weighted average. Masking must happen on logits, before softmax, so the softmax itself redistributes probability mass correctly over the unmasked positions.

**Drill 2 — Embedding mismatch that "looks" like a retrieval quality issue**
```python
# Indexing time
embeddings = model_a.encode(chunks)          # model_a
index.add(embeddings)

# Query time (different file, written weeks later)
query_vec = model_b.encode(query)            # model_b — different model!
results = index.search(query_vec, k=5)
```
*Bug:* two different embedding models used at index-time vs query-time. Their vector spaces aren't aligned, so similarity scores are meaningless even though the code runs without error and returns "results." This is the single most common silent RAG failure — always check that the encode call at query time is provably the same model/version/checkpoint as at index time.

**Drill 3 — LoRA that never learns**
```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base = base_layer  # <-- bug: base params never frozen
        in_dim, out_dim = base_layer.in_features, base_layer.out_features
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```
*Bug:* missing `for p in self.base.parameters(): p.requires_grad = False`. Without freezing, the optimizer updates the full base weight matrix as well as A/B — you're doing full fine-tuning with extra unnecessary parameters, not LoRA. Loss may look fine (it's still training something), but you lose LoRA's entire point: small trainable footprint, easy to swap adapters, fast to checkpoint. Symptom in practice: optimizer state / checkpoint size is much larger than expected for the given `r`.

**Drill 4 — Walk-forward CV that quietly leaks the future**
```python
def train_test_splits(df, n_splits=5):
    df = df.sample(frac=1).reset_index(drop=True)  # <-- bug: shuffles time order
    fold_size = len(df) // n_splits
    for i in range(n_splits):
        test_idx = df.index[i*fold_size:(i+1)*fold_size]
        train_idx = df.index.difference(test_idx)
        yield train_idx, test_idx
```
*Bug:* `df.sample(frac=1)` shuffles rows before splitting, destroying temporal order. This is random k-fold wearing a walk-forward costume — training folds can contain rows chronologically after the test fold, leaking future information. Backtest metrics will look great; production will not, because in production you never have future data. Fix: sort by timestamp, then split by an expanding or sliding time window, never shuffle.

**Drill 5 — WAPE computed in a way that hides the real error**
```python
def wape(actual, forecast):
    return np.mean(np.abs(actual - forecast) / actual)  # <-- bug: this is MAPE, not WAPE
```
*Bug:* this divides *inside* the mean, per-row — that's MAPE (and will blow up or divide-by-zero on any zero-demand SKU). Correct WAPE aggregates numerator and denominator **before** dividing:
```python
def wape(actual, forecast):
    return np.sum(np.abs(actual - forecast)) / np.sum(actual)
```
This is a good one to have memorized cold, since "explain why your WAPE implementation is actually WAPE and not MAPE with extra steps" is a realistic follow-up.

### Likely small coding asks
- Implement top-k / top-p (nucleus) sampling from logits
- Implement a simple token-bucket rate limiter for API calls
- Write a function to chunk text with overlap
- Implement BM25 scoring (or explain it conceptually + code the core TF-IDF-like formula)
- Implement a simple walk-forward CV splitter for a time series

---

## 20. System Design for LLM/RAG Applications

*A common prompt: "Design a customer-support chatbot for 1M users using our internal docs." Structure your answer like this:*

### 1. Clarify Requirements
- Scale (QPS, users), latency SLA, data freshness needs, budget constraints, accuracy/compliance requirements (PII? regulated industry?)

### 2. High-Level Architecture
```
User → API Gateway → [Cache check] → Router → 
   ├─ Simple query → Small/fast LLM (no retrieval)
   └─ Complex query → Retrieval (Vector DB + BM25 hybrid) 
                        → Rerank → Prompt Assembly → LLM → 
                        Guardrails/citation check → Response
                        ↳ Log to observability pipeline
```

### 3. Key Design Decisions to Narrate
- **Ingestion pipeline**: how docs get chunked/embedded/updated (batch nightly vs real-time streaming updates)
- **Vector DB choice**: justify based on scale + filtering needs (managed like Pinecone vs self-hosted like Qdrant)
- **Caching layer**: semantic cache for repeat queries; reduces cost/latency at scale
- **Fallback handling**: what happens if retrieval finds nothing relevant? (say "I don't know" rather than hallucinate)
- **Cost control**: model routing (cheap model for easy queries), fewer/reranked chunks, cache hit rate targets
- **Horizontal scaling**: stateless API layer behind load balancer; vector DB sharding for scale
- **Monitoring**: latency percentiles, retrieval hit rate, user feedback loop feeding back into eval set

### 4. Back-of-envelope Cost/Capacity Estimation (be ready to do simple math)
- Example: 1M users, 5 queries/day avg = 5M queries/day ≈ 58 QPS avg (much higher at peak)
- Estimate tokens per query (prompt + retrieved context + output) → estimate $/day using provider pricing
- Identify the bottleneck: usually LLM inference cost >> vector DB cost >> embedding cost at scale

### 5. Trade-offs to Mention Proactively (shows seniority)
- Accuracy vs latency vs cost — pick 2, justify the choice for the given use case
- Build vs buy (self-host OSS model vs API) — depends on scale, data privacy needs, ops maturity
- Freshness vs cost of re-indexing (real-time updates are expensive; batch is cheaper but stale)

---

## 21. Deploying Forecasting & RAG Solutions on AWS

*A very common "walk me through a production deployment" prompt — interviewers want to see you can name the right service for each stage, justify it against alternatives (including non-AWS/local options), and reason about latency, throughput, and cost, not just say "we used SageMaker for everything."*

### 20.1 Deploying a Demand Forecasting Solution on AWS

#### End-to-end architecture:

```
S3 (raw sales/transactions data)
   → Glue (ETL, data catalog, schema discovery)
   → Feature engineering (Glue/EMR/Athena, or SageMaker Processing jobs)
   → Model training:
        ├─ Amazon Forecast (managed AutoML: ARIMA/ETS/Prophet/DeepAR+/CNN-QR)
        └─ OR SageMaker Training Job (custom DeepAR / XGBoost / TFT)
   → SageMaker Model Registry (versioning, approval workflow)
   → Batch inference (SageMaker Batch Transform, or Amazon Forecast's built-in forecast generation)
   → Forecast output → S3 / Redshift / DynamoDB (for downstream consumption by planning systems)
   → Orchestration: Step Functions (or Managed Airflow / MWAA) chains ETL → train → evaluate → deploy
   → Scheduling/retraining trigger: EventBridge (e.g., weekly cron) → kicks off Step Functions pipeline
   → Monitoring: SageMaker Model Monitor (data/quality drift) + CloudWatch (pipeline health, job failures)
```

#### Key service choices and why

| Stage | AWS Service | Why |
|-------|-------------|-----|
| Raw data storage | **S3** | Cheap, durable, standard landing zone for structured + unstructured data |
| ETL / feature prep | **Glue** (serverless Spark) or **EMR** (more control) | Glue for simpler managed ETL; EMR when you need fine-grained Spark tuning at large scale |
| Managed forecasting | **Amazon Forecast** | AutoML across multiple algorithms (DeepAR+, CNN-QR, Prophet, ARIMA, ETS), handles cold-start and hierarchical data, minimal ML ops burden — good default unless you need full model control |
| Custom modeling | **SageMaker** (Training Jobs, built-in DeepAR algorithm, or bring-your-own container) | Needed when Forecast's algorithm set doesn't fit (custom TFT/N-BEATS, proprietary feature logic, tighter cost control at very large scale) |
| Batch inference | **SageMaker Batch Transform** | Demand forecasts are typically generated on a schedule (daily/weekly), not real-time — batch is cheaper and simpler than a hosted endpoint |
| Orchestration | **Step Functions** | Coordinates the multi-step pipeline (ETL → train → evaluate → publish) with retries/error handling; MWAA (managed Airflow) is the alternative if the team already standardizes on Airflow DAGs |
| Scheduling/retraining | **EventBridge** | Cron-based trigger for periodic retraining, decoupled from the pipeline logic itself |
| Forecast consumption | **Redshift** (analytics/BI) or **DynamoDB** (low-latency app lookups) | Depends on whether downstream consumer is a BI dashboard/planning tool or an application needing point lookups |
| Drift monitoring | **SageMaker Model Monitor** | Flags data quality drift (e.g., input distribution shift) that would silently degrade forecast accuracy |
| Security | **IAM roles (least privilege)**, **VPC endpoints** | Keep data access scoped per pipeline stage; avoid public internet egress for data in transit between services |

**Design notes worth mentioning in an interview:**
- Forecasting is inherently **batch/scheduled**, not real-time — this simplifies serving considerably vs an LLM RAG system (no need for low-latency hosted endpoints in most cases, so API Gateway/load balancing concerns barely apply here)
- Retraining cadence is a business tradeoff: too frequent = wasted compute + noisy model churn; too infrequent = stale model missing recent trend/promo shifts — typically weekly or monthly for demand forecasting
- Amazon Forecast vs custom SageMaker pipeline is a classic **build vs buy** tradeoff: Forecast gets you to production fast with good-enough accuracy; custom SageMaker pipelines are justified when you need probabilistic reconciliation across a hierarchy, proprietary features, or tighter cost control at very large SKU counts
- **Non-AWS/local alternative worth naming**: if you're not committed to AWS, the same pipeline maps to open-source tooling — Airflow/Prefect/Dagster for orchestration instead of Step Functions, a self-hosted Postgres/Parquet-on-object-storage instead of Redshift, and `statsforecast`/`neuralforecast`/`darts` (open-source Python libraries) running on plain EC2/on-prem GPUs or even a single beefy machine instead of Amazon Forecast. This is a legitimate answer when the interviewer asks "what if you weren't on AWS" — batch forecasting at moderate SKU counts (tens of thousands, not millions) doesn't strictly need managed services; it needs a scheduler and a place to write Parquet files. Managed AWS services buy you less ops burden, not capability you couldn't get otherwise.

### 20.2 Deploying a RAG Solution on AWS

#### End-to-end architecture:

```
Users → Route 53 (DNS) → CloudFront (CDN, optional, for static assets/UI)
   → Application Load Balancer (ALB) or API Gateway
        → API Gateway: request throttling, auth (Cognito/API keys), request validation, usage plans
        → ALB: needed instead of/alongside API Gateway when routing to long-lived
          ECS/Fargate services (agentic orchestration, streaming responses, WebSocket connections)
   → [Cache check: ElastiCache Redis — semantic cache for repeated/similar queries]
   → Compute layer:
        ├─ Lambda (simple retrieve-then-generate, short-lived, auto-scales per request)
        └─ ECS/Fargate (multi-step agentic RAG, longer-running orchestration, avoids Lambda's execution
                          time limit and cold-start hit on the hot path)
   → Retrieve top-k from vector store → (optional) rerank
   → Assemble prompt → Bedrock Runtime (invoke Claude/other model)
   → Bedrock Guardrails (PII/content filtering) → Response to user (streamed if supported)
   → Observability: CloudWatch (metrics/logs) + X-Ray (distributed tracing) + Bedrock invocation logging
```

**Document ingestion side (offline, feeds the vector store above):**
```
S3 (raw documents: PDFs, HTML, Confluence exports, etc.)
   → Ingestion & chunking (Lambda for small scale, or Glue jobs for large batch)
   → Embedding generation (Bedrock Titan Embeddings, or a SageMaker-hosted custom embedding model)
   → Vector store:
        ├─ OpenSearch Service (k-NN plugin) — self-managed control, hybrid (dense+BM25) search
        ├─ Amazon Kendra — managed enterprise semantic search, less retrieval tuning control
        ├─ Aurora PostgreSQL + pgvector — if already standardized on relational DBs
        └─ Bedrock Knowledge Bases — fully managed RAG (handles chunking + embedding + retrieval end-to-end)
```

#### API Gateway, load balancing, and latency — the parts that actually matter for RAG serving

- **API Gateway's job here** isn't just routing — it's the first line of defense for a public-facing RAG endpoint: request throttling (protects the LLM backend from being overwhelmed, since LLM inference is the expensive/slow part), API-key or Cognito-based auth, request/response validation (reject malformed payloads before they cost you an LLM call), and usage plans (per-customer rate limits/quotas for a multi-tenant product).
- **API Gateway vs ALB**: API Gateway is the default for simple request/response Lambda-backed RAG (its own scaling is fully managed, integrates natively with Lambda and Cognito). Once the RAG flow becomes agentic — multiple tool calls, streaming tokens back to the client, WebSocket connections for long-lived sessions — you outgrow API Gateway + Lambda's constraints (Lambda's ~15 min execution cap, and REST API Gateway doesn't stream well) and move to an **ALB in front of ECS/Fargate**, which supports long-lived connections, streaming HTTP responses, and WebSockets natively. A common real pattern: API Gateway for the public edge (auth, throttling) with a VPC Link forwarding to an internal ALB/ECS backend — you get API Gateway's edge features without losing ALB's connection flexibility.
- **Load balancing specifics**: ALB does layer-7 (HTTP-aware) routing — useful for path-based routing (`/chat` vs `/ingest` vs `/admin` to different target groups) and for weighted routing during canary deploys of a new prompt/model version. Health checks on the target group should hit a lightweight endpoint, not one that itself calls the LLM (a slow model call shouldn't look like an unhealthy node).
- **Where latency actually accumulates in a RAG request**, in the order it happens: (1) auth/throttle check at the gateway — sub-ms if done right; (2) semantic cache lookup — should be a few ms via ElastiCache, this is the highest-leverage latency win since it skips everything downstream; (3) embedding the query — small, fast if using a lightweight embedding model; (4) vector search — depends on index size and type (HNSW is sub-100ms typically at reasonable scale); (5) reranking, if used — adds real latency since it's a cross-encoder forward pass per candidate, so keep the reranked set small (top 20-50, not top 500); (6) **LLM generation itself — almost always the dominant cost**, split into time-to-first-token (driven by prompt length/prefill) and total generation time (driven by output length and decode speed). Streaming the response back to the client (rather than waiting for the full generation) is the single biggest perceived-latency win available and is worth explicitly designing for at the ALB/compute layer.
- **Horizontal scaling**: Lambda scales per-request automatically (up to account concurrency limits — worth knowing reserved/provisioned concurrency exists to avoid cold starts on latency-sensitive paths). ECS/Fargate scales via an Auto Scaling policy on the service, usually keyed to CPU/memory or a custom CloudWatch metric like in-flight request count — request count is a better scaling signal for LLM-backed services than CPU, since the bottleneck is usually an external API call (Bedrock), not local compute.
- **Cost/latency lever specific to Bedrock**: **Provisioned Throughput** (reserved capacity on Bedrock) trades a fixed cost for predictable low latency and no throttling under load — worth mentioning as the production answer to "what if on-demand Bedrock rate limits start throttling us during traffic spikes."

#### Key service choices and why

| Stage | AWS Service | Why |
|-------|-------------|-----|
| Document storage | **S3** | Standard landing zone, versioned, integrates natively with Bedrock Knowledge Bases as a data source |
| Chunking/embedding pipeline | **Lambda** (event-driven, small-medium scale) or **Glue** (large batch) | Lambda is simplest for incremental doc updates triggered by S3 events; Glue for large one-time/bulk backfills |
| Embeddings | **Bedrock Titan Embeddings** (managed) or custom model on **SageMaker endpoint** | Titan is the path of least resistance and stays inside the Bedrock ecosystem; custom SageMaker-hosted embeddings justified for a fine-tuned domain-specific embedding model |
| Vector store — build your own | **OpenSearch Service** | Most control: hybrid dense+BM25 search, custom index tuning, well-understood at scale |
| Vector store — managed enterprise search | **Kendra** | Good when you want managed relevance tuning and connectors to enterprise sources (SharePoint, Confluence) out of the box, less low-level control |
| Vector store — fully managed RAG | **Bedrock Knowledge Bases** | Simplest path: handles chunking, embedding, indexing (backed by OpenSearch Serverless or others), and retrieval in one managed service — good default unless you need very custom retrieval logic |
| Edge/API layer | **API Gateway** | Auth, throttling, usage plans, request validation — the right default for simple request/response RAG |
| Edge/API layer — agentic or streaming | **ALB** (+ VPC Link from API Gateway if you want both) | Long-lived connections, streaming, WebSockets — API Gateway alone struggles here |
| Query orchestration | **Lambda** (simple retrieve-then-generate) or **ECS/Fargate** (multi-step agentic RAG, longer-running orchestration) | Lambda has execution time limits and cold-start considerations — move to Fargate/ECS once the RAG flow becomes agentic (multiple tool calls, longer chains) |
| Generation | **Bedrock Runtime** (Claude, etc.) | Managed access to foundation models without hosting/serving infra yourself; **Provisioned Throughput** if you need guaranteed low latency at scale |
| Guardrails | **Bedrock Guardrails** | Built-in PII redaction, content filtering, denied-topics enforcement at the platform level |
| Caching | **ElastiCache (Redis)** | Semantic cache for repeated/similar queries — cuts cost and latency for FAQ-style traffic; also the single biggest latency lever in the whole pipeline when hit rate is decent |
| Auth | **Cognito** (user-facing) + **IAM** (service-to-service, least privilege) | Standard AWS-native auth split between end-user identity and internal service permissions |
| Observability | **CloudWatch + X-Ray** | Latency/error metrics plus distributed tracing across Lambda/API Gateway/Bedrock hops |
| Networking | **VPC + PrivateLink** | Keep Bedrock traffic off the public internet for regulated/enterprise deployments |

#### Non-AWS / local alternatives worth knowing (interviewers value knowing when NOT to reach for managed cloud services)

| AWS piece | Non-AWS / local / open-source alternative | When it's the better call |
|---|---|---|
| Bedrock Knowledge Bases / Bedrock Runtime | Self-hosted **vLLM** or **TGI (Text Generation Inference)** serving an open-weight model, with a self-hosted vector DB (Qdrant, Weaviate, Milvus, or plain FAISS for smaller scale) | Data residency/privacy requirements that rule out any managed cloud inference; very high, steady-state query volume where self-hosting is cheaper per-token than API calls; on-prem/air-gapped environments |
| OpenSearch Service | Self-hosted **Qdrant** or **Weaviate** (Docker/Kubernetes) | Smaller teams who want hybrid search without operating full OpenSearch clusters — Qdrant in particular is lighter-weight and easier to run locally/on a single VM for moderate scale |
| API Gateway + ALB | **Nginx** or **Traefik** as reverse proxy/load balancer, **Kong** or **Envoy** for API gateway features (rate limiting, auth) | Non-AWS deployment (on-prem, another cloud, or local Kubernetes) — these give you the same throttling/auth/routing capabilities without AWS lock-in |
| Step Functions / EventBridge | **Airflow**, **Prefect**, **Dagster**, or plain **cron** | Team already standardized on one of these, or the pipeline is simple enough that a managed orchestrator is overkill |
| ElastiCache (Redis) | Self-hosted **Redis** or **Memcached** | Identical functionality, just self-managed — makes sense once you're already self-hosting everything else and don't want a second cloud dependency |
| SageMaker Training/Batch Transform | Plain **EC2 GPU instances** running training scripts directly, or a local/on-prem GPU box for smaller models | Smaller-scale training jobs where SageMaker's orchestration overhead and per-job startup cost isn't worth it; teams comfortable managing their own training loop and checkpointing |
| CloudWatch + X-Ray | **Prometheus + Grafana** (metrics), **Jaeger** or **OpenTelemetry** (tracing), **Langfuse/Phoenix** (LLM-specific tracing) | Multi-cloud or on-prem setups, or when you specifically want LLM-native observability (token usage, prompt/completion logging) that generic CloudWatch doesn't give you out of the box |

**The honest framing for an interview**: managed AWS services buy you reduced *operational* burden (someone else runs the cluster, patches it, scales it) — they rarely buy you capability you couldn't get from open-source tooling. The right call depends on team size (small team → lean managed to avoid ops overhead), data sensitivity (regulated/on-prem → self-hosted), query volume economics (very high steady-state volume → self-hosting inference can undercut API pricing), and existing infra commitments (already all-in on Kubernetes elsewhere → self-hosted options fit better than adding a new AWS-specific service to the stack).

**Design notes worth mentioning in an interview:**
- **Bedrock Knowledge Bases vs "build your own with OpenSearch"** is the central build-vs-buy decision for RAG on AWS — Knowledge Bases gets you to production fast and is a strong default; a custom OpenSearch-based pipeline is justified when you need hybrid search tuning, custom rerankers, or non-standard chunking strategies Knowledge Bases doesn't support
- RAG serving is **real-time/low-latency**, unlike demand forecasting's batch nature — this drives the choice of Lambda/Fargate + API Gateway/ALB instead of batch jobs, and motivates the semantic caching layer
- Guardrails and PII handling matter more here than in forecasting, since RAG directly surfaces retrieved document content (and potentially generated text) to end users
- If asked to optimize an existing, too-slow RAG deployment, the actionable order of attack is usually: (1) add/improve semantic caching, (2) check if you're reranking too many candidates, (3) check if streaming is actually enabled end-to-end (gateway → compute → client), (4) check Bedrock on-demand throttling and consider Provisioned Throughput, (5) only then look at retrieval-side optimizations (smaller/better chunks, fewer results) — because generation latency usually dwarfs retrieval latency

### 20.3 Side-by-Side: Forecasting vs RAG Deployment

| Dimension | Forecasting Pipeline | RAG Pipeline |
|---|---|---|
| **Serving pattern** | Batch/scheduled (daily/weekly) | Real-time, low-latency |
| **Core managed service** | Amazon Forecast | Bedrock Knowledge Bases |
| **Custom-control alternative** | SageMaker Training + Batch Transform | OpenSearch Service + custom Lambda/Fargate orchestration |
| **Edge/entry point** | None needed (internal batch job) | API Gateway (simple) or ALB (streaming/agentic) |
| **Orchestration** | Step Functions / MWAA | API Gateway/ALB + Lambda/Fargate |
| **Trigger** | EventBridge (cron-based retraining) | User request (API call) |
| **Output store** | S3 / Redshift / DynamoDB | Returned directly to user (optionally logged to S3/CloudWatch) |
| **Key risk to monitor** | Data/concept drift degrading forecast accuracy | Retrieval quality drift, hallucination, stale index, latency spikes under load |
| **Primary AWS monitoring tool** | SageMaker Model Monitor | CloudWatch + X-Ray + Bedrock invocation logs |
| **Non-AWS fallback** | Airflow/Prefect + open-source forecasting libs on EC2/on-prem | Self-hosted vLLM/TGI + Qdrant/Weaviate behind Nginx/Envoy |

**Interview soundbite:** *"Forecasting on AWS is fundamentally a batch pipeline — Amazon Forecast or SageMaker Batch Transform on a scheduled trigger via EventBridge/Step Functions. RAG is fundamentally a real-time serving problem — Bedrock Knowledge Bases or a custom OpenSearch-backed retrieval layer behind API Gateway/ALB. The architectures differ mainly because one answers a question on a schedule and the other answers a question on demand, which is exactly why RAG needs to think hard about API Gateway throttling, load balancing, and streaming, and forecasting doesn't."*

---

## 22. Behavioral / Project Deep-Dive Prep

*Every AI Engineer interview has a "walk me through a project" component. Pre-think 2-3 projects using STAR (Situation, Task, Action, Result), specifically prepared for AI/ML framing.*

### Prompts to pre-answer for yourself
- "Walk me through a project where you improved retrieval/model quality — what was broken, how did you diagnose it, what did you change, what was the measurable impact?"
- "Tell me about a time a model/pipeline failed in production — how did you detect it, and what did you do?"
- "Describe a tradeoff you had to make between accuracy, latency, and cost — how did you decide?"
- "Tell me about a time you disagreed with a technical approach on your team — what happened?"

### Framing tips (specific to AI/ML projects)
- Always quantify impact where possible ("reduced hallucination rate from X% to Y%", "cut p95 latency from Xs to Ys", "reduced cost per query by X%", "improved WAPE from X% to Y%")
- Show the **diagnosis process**, not just the fix — interviewers care more about how you debug/reason than the final answer (this maps directly to your stated strength in debugging/analysis)
- If a project failed or an approach didn't work, be honest about it and explain what you learned/changed — this is often viewed more favorably than a suspiciously perfect story

---

## Quick Interview Soundbites (memorize these)

- *"Bias-variance tradeoff"* → underfitting vs overfitting, and how regularization/ensembling addresses each
- *"XGBoost uses second-order gradients (Hessian) for faster, more accurate splits than standard gradient boosting"*
- *"KV cache trades memory for compute — it's usually the actual bottleneck in LLM serving, not the model weights"*
- *"PagedAttention applies OS-style virtual memory paging to KV cache to eliminate fragmentation and enable memory sharing"*
- *"LoRA works because weight updates during fine-tuning have low intrinsic rank — you don't need to update all parameters to adapt a model"*
- *"QLoRA = 4-bit quantized frozen base + LoRA adapters trained in higher precision — enables fine-tuning huge models on single GPUs"*
- *"RAG optimization is fundamentally a precision/recall tradeoff at each stage — chunking, retrieval, reranking — and cost/latency tradeoff in production"*
- *"Faithfulness ≠ Relevance in RAG eval — an answer can be relevant to the question but not grounded in the retrieved context (hallucination), or grounded but not actually relevant"*
- *"LLM-as-judge is scalable but biased — always validate against a human-labeled sample first"*
- *"Transformers beat RNNs because self-attention is parallelizable across the sequence — the tradeoff is O(n²) attention cost, which is exactly what KV cache/FlashAttention address at inference"*
- *"An agent is RAG plus a decision loop — it decides whether, when, and how many times to retrieve or call tools, rather than a fixed single retrieval-then-generate pass"*
- *"Data parallelism scales throughput, tensor/pipeline parallelism scale model size — production-scale training usually combines all three"*
- *"When debugging a bad RAG pipeline, first check the boring stuff — embedding model mismatch between index-time and query-time is the most common silent failure"*
- *"DPO replaces RLHF's reward-model-plus-RL-loop with a single classification-style loss on preference pairs — same alignment goal, far simpler and more stable to train"*
- *"GRPO (Group Relative Policy Optimization) optimizes on group rankings instead of pairs — more sample-efficient when you have multiple outputs per prompt, used by reasoning-focused models"*
- *"Bi-encoders scale (independent encoding, pre-indexable) but cross-encoders are more accurate (joint attention) — that's exactly why retrieval uses bi-encoders and reranking uses cross-encoders"*
- *"Eval-driven development means the golden dataset is the CI gate for prompts and pipeline changes — the same way unit tests gate code changes, just with quality scores instead of pass/fail"*
- *"RoPE encodes relative position via rotation — better extrapolation than sinusoidal encoding, which is why LLaMA scales to long contexts"*
- *"GQA is grouped query attention — middle ground between full MHA (huge KV cache) and MQA (accuracy loss), increasingly common in production models"*
- *"Graph RAG trades chunking simplicity for structured knowledge + multi-hop reasoning — worth it when docs are dense with relationships"*
- *"Production systems know when they don't know — fallback handling (don't know, escalate, ask clarifying question) is as important as the happy path"*
- *"WAPE beats MAPE for demand forecasting — MAPE blows up on near-zero actuals from slow-moving SKUs, WAPE stays stable because it aggregates errors and actuals before dividing"*
- *"Time series CV must be walk-forward, never random k-fold — random splitting leaks future information into training"*
- *"Global forecasting models (DeepAR, TFT, feature-based XGBoost) beat per-series ARIMA at scale because they share learned patterns across thousands of related series and handle cold-start better"*
- *"Point forecasts aren't enough for inventory decisions — quantile/probabilistic forecasts (P50/P90) are what let you translate a forecast into a safety-stock and service-level decision"*
- *"Model selection in time series is a data-fit decision first, evaluation-score decision second: ARIMA for stationary/linear/short-horizon, LSTM for large non-linear/multivariate series with long dependencies, Prophet for seasonal data with missing values or holiday effects — and R² alone is a poor signal for time series because autocorrelated data already gives a naive baseline a head start"*
- *"In production RAG, the single biggest latency lever is almost always eliminating or caching work, not making retrieval faster — the optimization hierarchy is eliminate, parallelize, reduce, speed up, then improve perceived latency via streaming"*
- *"Forecasting on AWS is a batch pipeline (Amazon Forecast or SageMaker Batch Transform on an EventBridge-triggered schedule); RAG on AWS is a real-time serving problem (Bedrock Knowledge Bases or OpenSearch behind API Gateway/ALB) — the architectures diverge because one answers on a schedule, the other answers on demand"*
- *"Bedrock Knowledge Bases is the managed 'buy' option for RAG on AWS — handles chunking, embedding, and retrieval end-to-end; OpenSearch Service is the 'build' option when you need custom hybrid search or reranking logic"*
- *"In a RAG deployment, generation latency (the LLM call itself) almost always dwarfs retrieval latency — so the highest-leverage optimizations are semantic caching and streaming, not shaving milliseconds off vector search"*
- *"API Gateway is the right default for simple request/response RAG; once the flow becomes agentic or needs streaming/WebSockets, you move to ALB in front of ECS/Fargate — API Gateway alone doesn't handle long-lived connections well"*
- *"Managed cloud services buy you reduced ops burden, not capability — a self-hosted vLLM + Qdrant stack behind Nginx can do everything Bedrock + OpenSearch does, the decision comes down to data residency, query volume economics, and team size"*

---
