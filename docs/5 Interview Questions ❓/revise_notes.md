---
icon: lucide/briefcase
---

3. AI/ML Engineer Interview Revision Notes
   
### Table of Contents
1. Machine Learning Fundamentals
2. Deep Learning Basics & Transformers
3. Tokens, Embeddings & Position Encodings
4. LLM Architecture & Inference Parameters
5. LLM Training Pipeline: Pretraining → SFT → RLHF/DPO
6. Prompt Engineering & Few-Shot Learning
7. Agents & Tool Calling
8. Model Context Protocol (MCP)
9. Embeddings Deep-Dive
10. RAG (Retrieval-Augmented Generation)
11. Multimodal Models (Vision + Language)
12. Graph RAG & Knowledge Graphs
13. Latest LLM Models (July 2026)
14. Distributed Training & Inference
15. Observability & Evaluation
16. Testing & Eval-Driven Development for LLM Apps
17. Time Series Concepts
18. Demand Forecasting
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

## 2. Deep Learning Basics & Transformers

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

## 3. Tokens, Embeddings & Position Encodings

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

## 4. LLM Architecture & Inference Parameters

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

## 5. LLM Training Pipeline: Pretraining → SFT → RLHF/DPO

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

## 6. Prompt Engineering & Few-Shot Learning

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

## 7. Agents

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

## 8. Model Context Protocol (MCP)

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

## 9. Embeddings Deep-Dive

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

## 10. RAG (Retrieval-Augmented Generation)

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
- Latency/cost requirements (GPT-4o vs GPT-4o-mini, Claude Opus vs Haiku)
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
- E.g., simple factual query → small fast model (Haiku/mini); complex reasoning → larger model (Opus/GPT-4o)
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

## 11. Multimodal Models (Vision + Language)

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

### LLaVA, GPT-4V, Claude Vision
- **Architecture**: vision encoder (ViT or similar) → adapter/projection → language model
- **Process**: image → visual tokens → concatenate with text tokens → LLM processes unified sequence
- **Capabilities**: image understanding, OCR, visual reasoning, object counting, diagram interpretation
- **Limitation**: can hallucinate visual details or misinterpret complex scenes

### Multimodal Eval
- COCO Captions (caption quality), VQA (visual question answering accuracy), hallucination rate (false visual claims)
- Much harder to eval than text-only — often requires manual review or LLM-as-judge

---

## 12. Graph RAG & Knowledge Graphs

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

## 13. Latest LLM Models (July 2026)

*Know what's current — interviewers will reference new models, and "I haven't heard of that" hurts credibility.*

### Frontier Closed Models

#### **Claude (Anthropic)**
- **Claude Mythos 5**: Frontier tier, best overall on HLE (Humanity's Last Exam), reasoning-focused
- **Claude Opus 4.8**: Previous flagship, still very competitive, multimodal, 200K context
- **Claude Sonnet 5**: Mid-tier workhorse, great balance of speed/quality, 200K context, ~1-2M tok/s throughput
- **Claude Haiku 4.5**: Lightweight, fast, good for high-volume apps, 200K context
- **Claude Fable 5**: Frontier-tier with safety controls (limited biology, cybersecurity outputs); strongest on SWE-bench coding (95.0%)

**Key Anthropic innovation:** Constitutional AI + DPO alignment pipeline (no RLHF) → more stable, cheaper to train

#### **GPT-5 Family (OpenAI)**
- **GPT-5.4**: Flagship (March 2026), unified coding + general-purpose, **native computer use** (click, type, execute actions on screen)
- **GPT-5.6 variants** (Luna, Sol, Terra): Reasoning tiers — adaptive computation based on problem difficulty
- **GPT-5.6 Luna**: Lightweight variant, cost-optimized
- All GPT-5 models: 1M+ context window, multimodal (text, image, audio), ~500K-1M tok/s

**Key OpenAI innovation:** Configurable reasoning effort + inference-time scaling (similar to o1 reasoning model approach)

#### **Gemini 3.x Family (Google)**
- **Gemini 3.1 Pro**: Strong scientific reasoning (GPQA Diamond 94.3%), excellent multimodal (text, image, audio, video, PDF native)
- **Gemini 3.5 Flash** (July 2026): 4× speed improvement over 3.1, still high quality, best for throughput-critical apps
- **Gemini 3.6 Flash** (just released July 21, 2026): Latest iteration, fastest inference in frontier tier
- Context: 1M+ tokens, document retrieval benchmark leader

**Key Google innovation:** Native PDF, video, audio support in a single API call (Gemini 3.1 Pro+)

#### **Other Closed Models**
- **Grok 4.5** (xAI): Pure reasoning benchmark leader, cheapest in top-10 at $2/M tokens
- **MiniMax M3** (Moonshot AI): Strong on open-weight/closed-weight hybrid pricing

### Open-Weight Models (Self-Hostable)

| Model | Size | Key Strength | Context | Use Case |
|---|---|---|---|---|
| **Llama 4 Scout** | 405B | Longest context (10M tokens!) | 10M | Document-dense workloads, bulk retrieval |
| **DeepSeek V4 Pro** | 671B MoE | Reasoning + cost efficiency | 128K | Best open reasoning alternative |
| **Qwen 3.7 Max** | 405B | Near-frontier reasoning (92.4% GPQA) | 200K | Self-hosted reasoning |
| **GLM-5** | Variable | Strong coding + reasoning | 200K | Agentic workflows |
| **MiniMax M2.5** | ~405B | SWE-bench 80.2% (rivals GPT-5.4 on coding) | 200K | Coding-heavy self-hosted |
| **Kimi K3** | Unknown | Open weights promised July 27, 2026 | 200K+ | TBD (likely top-tier) |

**Key trend:** Open models have closed the coding gap — MiniMax M2.5 at 80.2% SWE-bench rivals frontier closed models. For self-hosted deployment, the decision is now **capability-per-dollar** vs API convenience, not pure capability.

### Model Selection Decision Tree (for interviews)

**Step 1: Privacy/deployment constraints?**
- Data cannot leave our infrastructure → open-weight (Llama 4, DeepSeek V4, Qwen)
- API use OK → proceed to Step 2

**Step 2: Task type?**
- Coding/agentic (SWE-bench heavy) → Claude Fable 5 or Llama 4 Scout (open) or DeepSeek V4 Pro (open)
- Scientific reasoning (GPQA heavy) → Gemini 3.1 Pro or Grok 4.5 or Qwen 3.7 Max
- Multimodal (video/audio/PDF) → Gemini 3.5 Flash or Gemini 3.1 Pro
- High-volume, latency-sensitive → GPT-5.6 Luna or Gemini 3.6 Flash (cheap + fast)
- Long-context retrieval (100K+) → Llama 4 Scout (10M) or Claude Opus (200K)

**Step 3: Latency budget?**
- <1s TTFT critical → Gemini 3.6 Flash (4× faster) or Claude Sonnet 5
- Moderate → Claude Opus or GPT-5.4
- Batch/offline → optimize for tokens/sec throughput

**Step 4: Cost per token?**
- High volume → Gemini 3.6 Flash or open-weight model
- Mixed budget → Claude Sonnet 5 or GPT-5.6 Luna (mid-tier pricing)
- Frontier quality → Claude Opus or GPT-5.4 (highest cost)

### Recent Innovations to Mention

#### **Reasoning Models (Inference-Time Scaling)**
- GPT-5 and Claude models now support adaptive reasoning — model can spend more compute on hard problems
- Trades latency for accuracy — useful for reasoning-heavy workloads, not for chat
- "Why it matters": pushing frontier from model scale to inference-time optimization instead

#### **Agentic AI as Default**
- All frontier models now support: tool use, planning, multi-step memory, computer use (GPT-5.4)
- Agents no longer require a separate scaffolding layer — baked into model behavior
- "Why it matters": interviewer will ask "how would you build an agent" — the answer is simpler now (just use the model directly with tool definitions)

#### **Multimodal Convergence**
- Gemini 3.1 Pro native support for text, image, audio, video, PDF in single API call
- No more separate vision encoder — end-to-end multimodal
- "Why it matters": RAG pipelines can now ingest videos/PDFs directly without preprocessing

#### **Context Window Explosion**
- Llama 4 Scout: 10M tokens (100K pages of text)
- GPT-5, Claude, Gemini all: 1M+ standard
- Context is no longer the bottleneck; **retrieval precision** is
- "Why it matters": chunking strategy matters less (can fit more docs), but retrieval quality matters more (don't get lost in the middle with 10M tokens)

### Benchmark Comparison (July 2026)

| Benchmark | Leader | Score | Runner-Up | Score |
|---|---|---|---|---|
| **GPQA Diamond** (hard reasoning) | Claude Mythos 5 | 94.6% | Gemini 3.1 Pro | 94.3% |
| **SWE-bench Verified** (coding) | Claude Fable 5 | 95.0% | Claude Opus 4.8 | 88.6% |
| **HLE** (all-around) | Claude Mythos 5 | 53.3% | Claude Fable 5 | 52.8% |
| **MRCR 1M** (document retrieval in 1M context) | Claude Opus 4.7 | 92.9% | Gemini 3.1 Pro | 92.1% |
| **ARC-AGI-2** (abstract reasoning) | Gemini 3.1 Pro | 77.1% | GPT-5.4 | 76.3% |

**Interview note:** Never cite a single benchmark — always mention task-specific winners. "Claude is best" is wrong; "Claude Fable 5 wins on SWE-bench but Gemini 3.1 Pro leads on scientific reasoning" is right.

### Pricing Context (as of July 2026)

| Model Tier | Input $/M tokens | Output $/M tokens | Typical Use |
|---|---|---|---|
| Frontier (Mythos, GPT-5.4) | $5-15 | $20-50 | Reasoning-heavy, low-volume |
| High (Opus, Sonnet, Gemini 3.1 Pro) | $1-3 | $5-15 | Production workloads, balanced |
| Mid (Claude Sonnet, GPT Luna, Gemini Flash) | $0.30-1 | $1-5 | High-volume, cost-sensitive |
| Lightweight (Haiku, GPT mini) | $0.05-0.30 | $0.15-1 | Bulk processing, RAG reranking |
| Open-weight self-hosted | $0 API | (your infra cost) | Privacy-critical, high volume |

**Interview angle:** "Cost-per-token is half the story. What matters is cost-per-task — a cheaper model might need twice as many retries or retrieval calls, flipping the total cost calculation."

### What to Say in Interviews

**"We started with GPT-4o but switched to Claude Sonnet 5 because..."**
- latency requirements (Sonnet is faster)
- better instruction-following for structured outputs
- agentic tool-use was more reliable
- 200K context let us retrieve more docs upfront (simpler pipeline)

**"We evaluated Llama 4 Scout internally because..."**
- 10M context = no chunking needed for large documents
- self-hosted = data privacy + cost at 100K+ QPS
- tradeoff: inference latency (slower than cloud API), ops burden

**"On the multimodal side, we chose Gemini 3.1 Pro because..."**
- native video support (no preprocessing)
- PDF handling built-in (no PDF extraction pipeline)
- scientific reasoning benchmark lead for physics/chemistry domain

**DON'T say:** "Claude is the best model" or "GPT-5 is always better." Interviewers know this is lazy thinking.

---

## 14. Distributed Training & Inference (surface-level, but expect it)

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

### Inference-side Scaling (ties back to Section 4)
- **Batching** (continuous batching) — throughput lever
- **Tensor parallelism at inference** — split a huge model across GPUs just to fit it in memory / lower latency
- **Quantization** — reduces per-GPU memory footprint, allows single-GPU serving
- **KV cache + PagedAttention** — the memory-efficiency lever specific to serving (see Section 4)

### Cost/Latency/Throughput Tradeoff (a favorite system-design thread)
- More GPUs / larger batch → higher throughput, but higher latency per request if batching too aggressively
- Time-to-first-token (TTFT) vs tokens/sec (generation speed) are usually optimized separately — TTFT dominated by prompt processing (prefill), generation speed dominated by decode step + KV cache reads

---

## 15. Observability & Evaluation

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
- **LLM-as-a-judge**: use a strong LLM (e.g., GPT-4o) to score outputs on rubrics (correctness, coherence, helpfulness) — scalable but has known biases (favors verbose answers, position bias in pairwise comparisons)
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

## 16. Testing & Eval-Driven Development for LLM Apps

*The natural bridge between "Evaluation" (Section 15) and real engineering practice — how do you actually prevent regressions in a non-deterministic system?*

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

## 17. Time Series Concepts

*Foundational for demand forecasting (Section 18) and for any "explain classical vs ML forecasting" interview question.*

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

**Interview trap:** MAPE breaks down badly on intermittent/low-volume demand data (division by near-zero actuals) — WAPE or MASE is the safer default for demand forecasting eval.

### Multi-Horizon Forecasting Strategies
- **Direct**: train a separate model per forecast horizon (h=1, h=2, ... h=n) — no error accumulation, but more models to maintain
- **Recursive**: forecast h=1, feed it back as input to forecast h=2, and so on — simple, but errors compound over the horizon
- **Multi-output**: single model predicts the entire horizon vector at once (common in DL models like TFT, N-BEATS) — captures cross-horizon dependencies directly

---

## 18. Demand Forecasting

*Applies Section 17's time series toolbox to the specific, very common business problem of predicting product/SKU-level demand for inventory and supply chain decisions.*

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

*A very common "walk me through a production deployment" prompt — interviewers want to see you can name the right AWS service for each stage and explain why, not just say "we used SageMaker for everything."*

### 21.1 Deploying a Demand Forecasting Solution on AWS

**End-to-end architecture:**
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

**Key service choices and why:**
| Stage | AWS Service | Why |
|---|---|---|
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
- Forecasting is inherently **batch/scheduled**, not real-time — this simplifies serving considerably vs an LLM RAG system (no need for low-latency hosted endpoints in most cases)
- Retraining cadence is a business tradeoff: too frequent = wasted compute + noisy model churn; too infrequent = stale model missing recent trend/promo shifts — typically weekly or monthly for demand forecasting
- Amazon Forecast vs custom SageMaker pipeline is a classic **build vs buy** tradeoff: Forecast gets you to production fast with good-enough accuracy; custom SageMaker pipelines are justified when you need probabilistic reconciliation across a hierarchy, proprietary features, or tighter cost control at very large SKU counts

### 21.2 Deploying a RAG Solution on AWS

**End-to-end architecture:**
```
S3 (raw documents: PDFs, HTML, Confluence exports, etc.)
   → Ingestion & chunking (Lambda for small scale, or Glue jobs for large batch)
   → Embedding generation (Bedrock Titan Embeddings, or a SageMaker-hosted custom embedding model)
   → Vector store:
        ├─ OpenSearch Service (k-NN plugin) — self-managed control, hybrid (dense+BM25) search
        ├─ Amazon Kendra — managed enterprise semantic search, less retrieval tuning control
        ├─ Aurora PostgreSQL + pgvector — if already standardized on relational DBs
        └─ Bedrock Knowledge Bases — fully managed RAG (handles chunking + embedding + retrieval end-to-end)
   → Query path: API Gateway → Lambda (or ECS/Fargate for heavier agent orchestration)
        → Retrieve top-k from vector store → (optional) rerank
        → Assemble prompt → Bedrock Runtime (invoke Claude/other model)
        → Bedrock Guardrails (PII/content filtering) → Response to user
   → Semantic cache: ElastiCache (Redis) for repeated/similar queries
   → Auth: Cognito (end users) + IAM (service-to-service)
   → Observability: CloudWatch (metrics/logs) + X-Ray (distributed tracing) + Bedrock invocation logging
   → CI/CD: CodePipeline + CodeBuild for automated deployment of Lambda/ECS services
   → Networking: VPC with PrivateLink to Bedrock for private connectivity, security groups scoping access
```

**Key service choices and why:**
| Stage | AWS Service | Why |
|---|---|---|
| Document storage | **S3** | Standard landing zone, versioned, integrates natively with Bedrock Knowledge Bases as a data source |
| Chunking/embedding pipeline | **Lambda** (event-driven, small-medium scale) or **Glue** (large batch) | Lambda is simplest for incremental doc updates triggered by S3 events; Glue for large one-time/bulk backfills |
| Embeddings | **Bedrock Titan Embeddings** (managed) or custom model on **SageMaker endpoint** | Titan is the path of least resistance and stays inside the Bedrock ecosystem; custom SageMaker-hosted embeddings justified for a fine-tuned domain-specific embedding model |
| Vector store — build your own | **OpenSearch Service** | Most control: hybrid dense+BM25 search, custom index tuning, well-understood at scale |
| Vector store — managed enterprise search | **Kendra** | Good when you want managed relevance tuning and connectors to enterprise sources (SharePoint, Confluence) out of the box, less low-level control |
| Vector store — fully managed RAG | **Bedrock Knowledge Bases** | Simplest path: handles chunking, embedding, indexing (backed by OpenSearch Serverless or others), and retrieval in one managed service — good default unless you need very custom retrieval logic |
| Query orchestration | **Lambda** (simple retrieve-then-generate) or **ECS/Fargate** (multi-step agentic RAG, longer-running orchestration) | Lambda has execution time limits and cold-start considerations — move to Fargate/ECS once the RAG flow becomes agentic (multiple tool calls, longer chains) |
| Generation | **Bedrock Runtime** (Claude, etc.) | Managed access to foundation models without hosting/serving infra yourself |
| Guardrails | **Bedrock Guardrails** | Built-in PII redaction, content filtering, denied-topics enforcement at the platform level |
| Caching | **ElastiCache (Redis)** | Semantic cache for repeated/similar queries — cuts cost and latency for FAQ-style traffic |
| Auth | **Cognito** (user-facing) + **IAM** (service-to-service, least privilege) | Standard AWS-native auth split between end-user identity and internal service permissions |
| Observability | **CloudWatch + X-Ray** | Latency/error metrics plus distributed tracing across Lambda/API Gateway/Bedrock hops |
| Networking | **VPC + PrivateLink** | Keep Bedrock traffic off the public internet for regulated/enterprise deployments |

**Design notes worth mentioning in an interview:**
- **Bedrock Knowledge Bases vs "build your own with OpenSearch"** is the central build-vs-buy decision for RAG on AWS — Knowledge Bases gets you to production fast and is a strong default; a custom OpenSearch-based pipeline is justified when you need hybrid search tuning, custom rerankers, or non-standard chunking strategies Knowledge Bases doesn't support
- RAG serving is **real-time/low-latency**, unlike demand forecasting's batch nature — this drives the choice of Lambda/Fargate + API Gateway instead of batch jobs, and motivates the semantic caching layer
- Guardrails and PII handling matter more here than in forecasting, since RAG directly surfaces retrieved document content (and potentially generated text) to end users

### 21.3 Side-by-Side: Forecasting vs RAG Deployment

| Dimension | Forecasting Pipeline | RAG Pipeline |
|---|---|---|
| **Serving pattern** | Batch/scheduled (daily/weekly) | Real-time, low-latency |
| **Core managed service** | Amazon Forecast | Bedrock Knowledge Bases |
| **Custom-control alternative** | SageMaker Training + Batch Transform | OpenSearch Service + custom Lambda/Fargate orchestration |
| **Orchestration** | Step Functions / MWAA | API Gateway + Lambda/Fargate |
| **Trigger** | EventBridge (cron-based retraining) | User request (API call) |
| **Output store** | S3 / Redshift / DynamoDB | Returned directly to user (optionally logged to S3/CloudWatch) |
| **Key risk to monitor** | Data/concept drift degrading forecast accuracy | Retrieval quality drift, hallucination, stale index |
| **Primary AWS monitoring tool** | SageMaker Model Monitor | CloudWatch + X-Ray + Bedrock invocation logs |

**Interview soundbite:** *"Forecasting on AWS is fundamentally a batch pipeline — Amazon Forecast or SageMaker Batch Transform on a scheduled trigger via EventBridge/Step Functions. RAG is fundamentally a real-time serving problem — Bedrock Knowledge Bases or a custom OpenSearch-backed retrieval layer behind API Gateway/Lambda. The architectures differ mainly because one answers a question on a schedule and the other answers a question on demand."*

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
- *"Frontier model choice is task-specific, not all-purpose — Claude Fable 5 leads on coding, Gemini 3.1 Pro leads on scientific reasoning, they trade off differently by domain"*
- *"Context window explosion (Llama 4 at 10M) flips the tradeoff — chunking matters less now, retrieval precision matters more (lost in the middle with that much context)"*
- *"Open-weight models have closed the coding gap (MiniMax M2.5 at SWE-bench 80.2%) — decision is now capability-per-dollar self-hosted vs API convenience, not pure capability"*
- *"Agentic AI is no longer a framework question — frontier models now natively support tool use, planning, multi-step memory. Just define tool schemas and let the model decide"*
- *"Multimodal is converged (Gemini 3.1 Pro native text/image/audio/video/PDF) — RAG pipelines no longer need separate vision encoders or PDF extraction"*
- *"Inference-time reasoning scaling (GPT-5, Claude) trades latency for accuracy — useful for reasoning-heavy workloads, not for chat. Know when to trade off"*
- *"MCP (Model Context Protocol) standardizes how LLMs connect to tools/data sources — moves from inline tool definitions to composable, versioned tool servers"*
- *"Unlike function calling where tool code is embedded in your app, MCP servers are independent microservices that multiple Claude apps can use"*
- *"MCP is for production systems with multiple apps needing the same tools — enables code reuse, independent updates, and team-owned tool ownership"*
- *"MCP supports multiple transports (stdio, HTTP, WebSocket) — start with local subprocess, scale to distributed services without changing tool code"*
- *"WAPE beats MAPE for demand forecasting — MAPE blows up on near-zero actuals from slow-moving SKUs, WAPE stays stable because it aggregates errors and actuals before dividing"*
- *"Time series CV must be walk-forward, never random k-fold — random splitting leaks future information into training"*
- *"Global forecasting models (DeepAR, TFT, feature-based XGBoost) beat per-series ARIMA at scale because they share learned patterns across thousands of related series and handle cold-start better"*
- *"Point forecasts aren't enough for inventory decisions — quantile/probabilistic forecasts (P50/P90) are what let you translate a forecast into a safety-stock and service-level decision"*
- *"Forecasting on AWS is a batch pipeline (Amazon Forecast or SageMaker Batch Transform on an EventBridge-triggered schedule); RAG on AWS is a real-time serving problem (Bedrock Knowledge Bases or OpenSearch behind API Gateway/Lambda) — the architectures diverge because one answers on a schedule, the other answers on demand"*
- *"Bedrock Knowledge Bases is the managed 'buy' option for RAG on AWS — handles chunking, embedding, and retrieval end-to-end; OpenSearch Service is the 'build' option when you need custom hybrid search or reranking logic"*

---
