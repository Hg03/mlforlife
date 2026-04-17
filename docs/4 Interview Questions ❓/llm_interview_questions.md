---
icon: lucide/folder-symlink
---

# 2. LLM Interview Questions

### 1. What does tokenization entail, and why is it critical for LLMs?
**Tokenization** is the process of breaking down raw text into smaller units called **tokens**, which can be words, subwords, or individual characters. For example, "artificial" might be split into "art," "ific," and "ial."
* **Criticality**: LLMs process numerical representations of tokens, not raw text.
* **Benefits**: It enables models to handle diverse languages, manage rare words, and optimize vocabulary size for computational efficiency.

### 2. How does the attention mechanism function in transformer models?
The attention mechanism allows LLMs to weigh the importance of different tokens in a sequence when generating or interpreting text.
* **Mechanism**: It computes similarity scores between **query, key, and value vectors** using operations like dot products.
* **Example**: In "The cat chased the mouse," attention helps the model specifically link "mouse" to "chased" to understand the context.

### 3. What is the context window in LLMs, and why does it matter?
The context window refers to the number of tokens an LLM can process at once, defining its "memory" for a single task.
* **Impact**: A larger window (e.g., 32,000 tokens) allows the model to consider more context, improving coherence in long-form tasks like document summarization.
* **Trade-off**: Larger windows increase computational costs.

### 4. What distinguishes LoRA from QLoRA in fine-tuning LLMs?
* **LoRA (Low-Rank Adaptation)**: A fine-tuning method that adds low-rank matrices to a model's layers, enabling efficient adaptation with minimal memory overhead.
* **QLoRA**: Extends LoRA by applying **quantization** (e.g., 4-bit precision) to further reduce memory usage. It can fine-tune a 70B-parameter model on a single GPU.

### 5. How does beam search improve text generation compared to greedy decoding?
* **Greedy Decoding**: Selects only the single most probable word at each step.
* **Beam Search**: Explores multiple sequences simultaneously, keeping the top *k* candidates (beams). This ensures more coherent outputs by balancing probability and diversity.

### 6. What role does temperature play in controlling LLM output?
Temperature is a hyperparameter that adjusts the randomness of token selection.
* **Low Temperature (e.g., 0.3)**: Favors high-probability tokens, producing predictable, focused outputs.
* **High Temperature (e.g., 1.5)**: Increases diversity by flattening the probability distribution.
* **Balanced**: 0.8 is often the "sweet spot" for storytelling.

### 7. What is masked language modeling, and how does it aid pretraining?
**Masked Language Modeling (MLM)** involves hiding random tokens in a sequence and training the model to predict them based on surrounding context.
* **Benefit**: Used in models like BERT, MLM fosters bidirectional understanding, allowing the model to grasp complex semantic relationships.

### 8. What are sequence-to-sequence models, and where are they applied?
**Seq2Seq** models transform an input sequence into an output sequence, often of different lengths.
* **Architecture**: They consist of an **encoder** to process the input and a **decoder** to generate the output.
* **Applications**: Machine translation, text summarization, and chatbots.

### 9. How do autoregressive and masked models differ in LLM training?
* **Autoregressive Models (e.g., GPT)**: Predict tokens sequentially based on prior tokens. They excel in generative tasks like text completion.
* **Masked Models (e.g., BERT)**: Predict masked tokens using bidirectional context, making them ideal for comprehension tasks like classification.

### 10. What are embeddings, and how are they initialized in LLMs?
Embeddings are dense vectors that represent tokens in a continuous space, capturing semantic and syntactic properties.
* **Initialization**: They are often initialized randomly or with pretrained models like GloVe, then fine-tuned during training to reflect specific task contexts.

### 11. What is next sentence prediction, and how does it enhance LLMs?
**Next Sentence Prediction (NSP)** trains models to determine if two sentences are consecutive or unrelated. During pretraining, models like BERT learn to classify sequential vs. random sentence pairs, which improves coherence in dialogue and summarization.

### 12. How do top-k and top-p sampling differ in text generation?
* **Top-k Sampling**: Selects from the *k* most probable tokens, ensuring controlled diversity.
* **Top-p (Nucleus) Sampling**: Chooses tokens whose cumulative probability exceeds a threshold *p* (e.g., 0.95). It offers more flexibility by adapting to the context's probability distribution.

### 13. Why is prompt engineering crucial for LLM performance?
Prompt engineering involves designing inputs to elicit specific desired responses. A clear prompt (e.g., "Summarize in 100 words") improves output relevance and enables LLMs to tackle tasks in zero-shot or few-shot settings without extensive retraining.

### 14. How can LLMs avoid catastrophic forgetting during fine-tuning?
**Catastrophic forgetting** occurs when fine-tuning erases prior knowledge. Mitigation strategies include:
* **Rehearsal**: Mixing old and new data during training.
* **Elastic Weight Consolidation**: Prioritizing critical weights to preserve knowledge.
* **Modular Architectures**: Adding task-specific modules rather than overwriting existing weights.

### 15. What is model distillation, and how does it benefit LLMs?
Model distillation trains a smaller **"student"** model to mimic a larger **"teacher"** model's outputs using soft probabilities. This reduces memory and compute requirements, enabling deployment on mobile devices while retaining near-teacher performance.

### 16. How do LLMs manage out-of-vocabulary (OOV) words?
LLMs use **subword tokenization** (like Byte-Pair Encoding). For instance, "cryptocurrency" might split into "crypto" and "currency." This allows the model to process rare or new words using known sub-units.

### 17. How do transformers improve on traditional Seq2Seq models?
Transformers overcome limitations through:
* **Parallel Processing**: Self-attention allows simultaneous token processing, unlike sequential RNNs.
* **Long-Range Dependencies**: Attention captures relationships between distant tokens easily.
* **Positional Encodings**: These preserve the sense of sequence order without needing sequential steps.

### 18. What is overfitting, and how can it be mitigated in LLMs?
Overfitting occurs when a model memorizes training data and fails to generalize. Mitigation includes **Regularization** ($L1/L2$ penalties), **Dropout** (disabling neurons randomly), and **Early Stopping** (halting training when validation performance plateaus).

### 19. What are generative versus discriminative models in NLP?
* **Generative Models (e.g., GPT)**: Model joint probabilities to create new data (text or images).
* **Discriminative Models (e.g., BERT)**: Model conditional probabilities to distinguish between classes (e.g., sentiment analysis).

### 20. How does GPT-4 differ from GPT-3 in features and applications?
GPT-4 introduced:
* **Multimodal Input**: Processes both text and images.
* **Larger Context**: Up to 25,000 tokens (vs GPT-3's 4,096).
* **Enhanced Accuracy**: Significant reduction in factual errors.

### 21. What are positional encodings, and why are they used?
Self-attention lacks inherent order awareness. Positional encodings add information about the sequence order to the input (using sinusoidal functions or learned vectors), ensuring the model understands the relative position of words.

### 22. What is multi-head attention, and how does it enhance LLMs?
Multi-head attention splits queries, keys, and values into multiple subspaces. This allows the model to focus on different aspects of the input simultaneously (e.g., one head for syntax, another for semantics).

### 23. How is the softmax function applied in attention mechanisms?
The **softmax** function normalizes attention scores into a probability distribution:
$$softmax(x_{i})=\frac{e^{x_{i}}}{\sum_{j}e^{x_{j}}}$$
In attention, it converts raw similarity scores into weights that emphasize relevant tokens.

### 24. How does the dot product contribute to self-attention?
The dot product between query ($Q$) and key ($K$) vectors computes similarity:
$$Score=\frac{Q\cdot K}{\sqrt{d_{k}}}$$
High scores indicate relevant tokens. While efficient, its complexity is quadratic ($O(n^{2})$).

### 25. Why is cross-entropy loss used in language modeling?
It measures the divergence between predicted and true token probabilities:
$$L=-\sum y_{i}log(\hat{y}_{i})$$
It penalizes incorrect predictions, forcing the model to assign high probabilities to the correct next tokens.

### 26. How are gradients computed for embeddings in LLMs?
Gradients are computed using the **chain rule** during backpropagation. These gradients adjust the embedding vectors to minimize loss, refining their semantic representation.

### 27. What is the Jacobian matrix's role in transformer backpropagation?
The Jacobian matrix captures partial derivatives of outputs with respect to inputs. It helps compute gradients for multidimensional outputs, ensuring accurate weight updates in complex transformer architectures.

### 28. How do eigenvalues and eigenvectors relate to dimensionality reduction?
Eigenvectors define principal directions in data, and eigenvalues indicate their variance. In techniques like **PCA**, selecting eigenvectors with high eigenvalues reduces dimensionality while retaining the most important variance.

### 29. What is KL divergence, and how is it used in LLMs?
**KL divergence** quantifies the difference between two probability distributions:
$$D_{KL}(P||Q)=\sum P(x)log\frac{P(x)}{Q(x)}$$
In LLMs, it is used to evaluate how closely model predictions match true distributions during fine-tuning.

### 30. What is the derivative of the ReLU function, and why is it significant?
The derivative is 1 if $x > 0$, and 0 otherwise. Its sparsity and non-linearity prevent **vanishing gradients**, making ReLU computationally efficient and robust for deep LLM training.

### 31. How does the chain rule apply to gradient descent in LLMs?
The chain rule enables backpropagation by calculating derivatives of composite functions layer by layer. This allows the model to update parameters in deep architectures to minimize loss efficiently.

### 32. How are attention scores calculated in transformers?
Attention scores are computed as:
$$Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$$
The scaled dot product measures relevance, and softmax normalizes it to focus on key tokens.

### 33. How does Gemini optimize multimodal LLM training?
Gemini uses a **Unified Architecture** to combine text and image processing, **Advanced Attention** for cross-modal stability, and **Data Efficiency** via self-supervised techniques.

### 34. What types of foundation models exist?
* **Language Models**: BERT, GPT-4.
* **Vision Models**: ResNet.
* **Generative Models**: DALL-E.
* **Multimodal Models**: CLIP.

### 35. How does PEFT mitigate catastrophic forgetting?
**Parameter-Efficient Fine-Tuning (PEFT)** updates only a small subset of parameters (e.g., via LoRA) while freezing the rest. This ensures the model adapts to new tasks without losing core pretrained knowledge.

### 36. What are the steps in Retrieval-Augmented Generation (RAG)?
1. **Retrieval**: Fetching documents using query embeddings.
2. **Ranking**: Sorting documents by relevance.
3. **Generation**: Using the retrieved context to generate an accurate response.

### 37. How does Mixture of Experts (MoE) enhance LLM scalability?
MoE uses a **gating function** to activate only specific "expert" sub-networks for a given input. This allows a model to have billions of parameters while only using a fraction (e.g., 10%) per query, reducing computational load.

### 38. What is Chain-of-Thought (CoT) prompting, and how does it aid reasoning?
CoT prompting guides LLMs to solve problems step-by-step. For math or logic, it breaks down the process into intermediate calculations, improving both accuracy and interpretability.

### 39. How do discriminative and generative AI differ?
* **Discriminative AI**: Predicts labels based on input features (conditional probabilities).
* **Generative AI**: Creates new data by modeling the underlying distribution (joint probabilities).

### 40. How does knowledge graph integration improve LLMs?
Knowledge graphs provide structured, factual data. They help **reduce hallucinations**, improve reasoning through entity relationships, and offer better context for question-answering tasks.

### 41. What is zero-shot learning, and how do LLMs implement it?
Zero-shot learning allows LLMs to perform untrained tasks using general knowledge from pretraining. For example, a model can infer how to classify a review as "positive" just from the prompt instructions.

### 42. How does Adaptive Softmax optimize LLMs?
It groups words by frequency, using fewer computations for rare words. This lowers the cost of handling massive vocabularies, speeding up training and inference in resource-limited settings.

### 43. How do transformers address the vanishing gradient problem?
They use **Self-Attention** (avoiding sequential bottlenecks), **Residual Connections** (allowing direct gradient flow), and **Layer Normalization** (stabilizing updates).

### 44. What is few-shot learning, and what are its benefits?
Few-shot learning enables LLMs to perform tasks with only a handful of examples. Benefits include reduced data needs, faster adaptation, and cost efficiency for niche tasks.

### 45. How would you fix an LLM generating biased or incorrect outputs?
1. **Analyze Patterns**: Identify bias sources in data.
2. **Enhance Data**: Use balanced, debiased datasets.
3. **Fine-Tune**: Retrain with curated data or adversarial methods.

### 46. How do encoders and decoders differ in transformers?
**Encoders** process input into abstract representations capturing context. **Decoders** generate outputs using the encoder's data and previously generated tokens. In translation, the encoder "understands" the source and the decoder "writes" the target.

### 47. How do LLMs differ from traditional statistical language models?
Traditional models (like **N-grams**) rely on simpler, supervised methods. LLMs use transformer architectures, massive datasets, and unsupervised pretraining to handle long-range context and diverse tasks.

### 48. What is a hyperparameter, and why is it important?
Hyperparameters (like learning rate or batch size) are preset values that control the training process. They influence how fast a model converges and its final accuracy.

### 49. What defines a Large Language Model (LLM)?
LLMs are AI systems trained on vast text corpora to understand and generate human-like language. They typically feature billions of parameters and excel at translation, summarization, and reasoning.

### 50. What challenges do LLMs face in deployment?
Major challenges include **Resource Intensity** (high compute demand), **Bias** (perpetuating training data flaws), **Interpretability** (difficulty in explaining outputs), and **Privacy** (data security concerns).