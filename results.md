# Evaluation 

## Q1: What is Self-Attention
- **Expected Answer:** Self-attention is also known as intra-attention where the input sequence is compared to itself. Each element of the sequence interacts with all other elements in the sequence to compute dependencies between them.
- **Model Answer:** Self‑Attention, also known as intra‑attention, is an attention mechanism that relates different positions within a single sequence to compute a representation of that sequence.
- **Correct?** Yes
- **Comments:** Strong conceptual match; missing explicit mention of interactions and dependencies.

## Q2: What is Machine Learning
- **Expected Answer:** Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn from data, identify patterns, and make decisions or predictions without being explicitly programmed for every specific rule
- **Model Answer:** Machine learning is a subset of artificial intelligence that can learn automatically from experience and modify itself without needing external programming.
- **Correct?** Yes
- **Comments:** Good match; slightly less emphasis on data and pattern identification.

## Q3: What is NLP
- **Expected Answer:** Natural Language Processing (NLP) is a branch of artificial intelligence (AI) that enables computers to understand, interpret, and generate human language.
- **Model Answer:**Natural Language Processing (NLP) is the computational processing of human language text. It encompasses a range of tasks such as tokenization, classification, stemming, part‑of‑speech tagging, parsing, named‑entity recognition, sentiment analysis, and other linguistic analyses applied to written language.
- **Correct?** Yes
- **Comments:** Accurate and detailed, but slightly misses explicit mention of language generation.

## Q4: What is role of PyTorch in Python
- **Expected Answer:** PyTorch is an open-source Python-based deep learning framework that provides a flexible platform for building, training, and deploying machine learning models, especially neural networks
- **Model Answer:** PyTorch serves as a **Python‑centric machine‑learning library** that lets users write models in ordinary, imperative Python code while still getting high‑performance execution (including GPU acceleration).  It integrates tightly with the Python ecosystem—offering simple, consistent, “Pythonic” interfaces for tensors, operators, and automatic differentiation—so that data‑scientists can develop, debug, and run models just like any other Python program.  
- **Comments:** Correct with more implementation-focused details.

## Q5: What is the role of embeddings in NLP Embeddings
- **Expected Answer:**Embeddings in NLP convert text into dense, numerical vectors, allowing models to understand semantic meaning and relationships between words.
- **Model Answer:** Embeddings serve as dense vector representations that capture the semantic and syntactic properties of language units (words, sentences, paragraphs) and are used as the foundational input to modern NLP models.  By converting discrete tokens into continuous vectors, they allow neural architectures to process text more effectively, enable transfer‑learning across tasks, and consistently improve performance compared with learning representations from scratch.  In models such as BERT, the final input representation is the sum of token embeddings, segment embeddings, and position embeddings, illustrating how embeddings are integral to building contextualized language representations.  
- **Correct?** Yes
- **Comments:** Very strong match with additional detail.

## Q6: What is the connection between BERT and Attention
- **Expected Answer:**BERT (Bidirectional Encoder Representations from Transformers) is fundamentally built upon the Transformer encoder architecture, using multi-head self-attention mechanisms to process text. 
- **Model Answer:** BERT’s core architecture is built on the Transformer’s **self‑attention** mechanism.  
Instead of using separate stages for encoding each sentence and then applying a cross‑attention step (as in earlier models such as Parikh et al., 2016; Seo et al., 2017), BERT **unifies** these stages by feeding a concatenated pair of sentences into a **bidirectional self‑attention** layer. This allows every token to attend to both its left and right context at every layer, enabling the model to capture interactions within a single text and across text pairs directly.
- **Correct?** Yes
- **Comments:** Accurate and slightly more detailed.

## Q7: How do Deep Learning Libraries and NLP models interact in real applications.
- **Expected Answer:** Deep learning libraries provide the foundational tools and infrastructure for building, training, and deploying sophisticated NLP models in real-world applications. They enable models to handle the complexity and nuance of human language by leveraging neural networks and large-scale data
- **Model Answer:** Deep‑learning libraries (e.g., PyTorch) supply the programming‑level building blocks—Python classes for layers, model containers, and efficient runtime kernels—that let researchers and engineers instantiate, train, and run NLP architectures such as BERT.  In practice, a pre‑trained NLP model is packaged as a set of parameters that can be loaded into a library‑defined model class; the library then handles forward‑propagation, gradient computation, and hardware acceleration while abstracting away low‑level details like the Python Global Interpreter Lock (GIL) and static‑graph construction (Source: Performance focused implementation, Page 5).  Because these libraries expose a uniform, stateful‑function API, new or custom NLP components can be added simply by defining new layer classes and composing them into a model, making it straightforward to adapt or extend models like BERT for downstream tasks 
- **Correct?** Yes
- **Comments:** Correct but slightly more detailed.

## Q8: What is Pre-training BERT
- **Expected Answer:** Pre-training BERT is the initial, unsupervised learning phase where the model learns language representations from vast amounts of unlabeled text
- **Model Answer:** Pre‑training BERT is the first stage of the BERT framework in which the model is trained on large amounts of unlabeled text.  Rather than using a traditional left‑to‑right or right‑to‑left language model, BERT is trained with two unsupervised objectives (e.g., masked‑language modeling and next‑sentence prediction) on token sequences whose input representation is built by summing token, segment and position embeddings.  This pre‑training step learns the [CLS], [SEP] and sentence‑A/B embeddings before the model is later fine‑tuned on downstream tasks.
- **Correct?** Yes
- **Comments:** Highly accurate and detailed.

## Q9: Define Multiprocessing
- **Expected Answer:** multiprocessing is the ability to run multiple tasks simultaneously by leveraging multiple CPU cores or GPUs, using the torch.multiprocessing module, which is a wrapper around Python's built-in multiprocessing library.
- **Model Answer:** Multiprocessing is a Python standard module that provides utilities for spawning child processes and implementing basic inter‑process communication primitives, enabling parallel execution of code to work around the Global Interpreter Lock (GIL).
- **Correct?** Yes
- **Comments:** Conceptually correct about parallel execution and GIL, but misses torch.multiprocessing and GPU context.

## Q10: What is Embeddings & Softmax
- **Expected Answer:** Embeddings and Softmax are fundamental components in neural networks for processing data and producing outputs.
- **Model Answer:** **Embeddings** are vector representations that encode the information of each input token. In the model described, each token’s final representation is obtained by summing three kinds of embeddings:

* **Token embeddings** – learned vectors for the individual WordPiece tokens.  
* **Segment embeddings** – vectors that indicate which sentence (or segment) a token belongs to.  
* **Position embeddings** – vectors that encode the token’s position in the sequence.

These summed embeddings are then fed into the transformer layers. The same weight matrix is also shared between the token‑embedding layers and the pre‑softmax linear transformation, with the weights scaled by √dₘₒ𝒹ₑₗ (Source: provided context, n/a).

**Softmax** is the activation function applied after the model produces raw logits (the output of a linear layer). It converts those logits into a probability distribution over the vocabulary, allowing the model to predict the next token (or a masked token in a masked‑language‑model task). In the described architecture, the decoder output is passed through a softmax to obtain predicted next‑token probabilities, and the hidden vectors for masked tokens are fed into a softmax over the vocabulary during MLM training 
- **Correct?** Yes
- **Comments:** Correct with strong technical explanation.
