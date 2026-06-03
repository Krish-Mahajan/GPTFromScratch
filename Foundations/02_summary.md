# Notebook 02: Neural Language Models and Word Embeddings

## Core Idea

Replace count tables with learned parameters. Words become dense vectors (embeddings) in continuous space — similar words get similar vectors, so knowledge transfers automatically.

## Key Concepts

### Word Embeddings

Instead of treating each word as an arbitrary index, represent it as a vector of real numbers:

```
"cat" = [0.2, 0.8, -0.1, 0.5, ...]
"dog" = [0.3, 0.7, -0.2, 0.4, ...]  ← close to cat!
```

The embedding matrix C (shape: vocab_size × embed_dim) is learned during training. Each row is a word's vector.

### Bengio's Neural Language Model (2003)

The architecture that started neural NLP. Four steps:

1. **Embedding lookup**: context word indices → dense vectors via matrix C
2. **Concatenate**: join context embeddings into one flat vector
3. **Hidden layer**: linear transform + tanh activation (learns non-linear patterns)
4. **Output layer**: linear transform → logits over vocabulary, softmax → probabilities

```
P(next_word | context) = softmax(W × tanh(H × [C(w1); C(w2)] + d) + b)
```

Loss = cross-entropy (negative log-likelihood of the correct next word).

### Word2Vec (Skip-gram)

Flips the prediction: given a center word, predict surrounding context words.

- Two embedding tables: center_embeddings and context_embeddings
- Score = dot product of center and context vectors
- Training uses negative sampling: real pairs → label 1, random pairs → label 0
- Binary cross-entropy loss pushes real pairs together, fake pairs apart

Result: words sharing similar contexts end up with similar embeddings.

### Cosine Similarity

Measures how similar two word vectors are (ignoring magnitude):

```
cosine(u, v) = (u · v) / (||u|| × ||v||)
```

Range: -1 (opposite) to +1 (identical direction). Used to verify embeddings learned semantic similarity.

### RNN Language Model

Processes words one at a time, carrying a hidden state that accumulates context:

```
h_t = tanh(W_xh × embed(word_t) + W_hh × h_{t-1})
prediction_t = W_output × h_t
```

Key difference from Bengio: no fixed context window. The hidden state theoretically encodes the entire history.

Data format: input = sentence[:-1], target = sentence[1:] (predict next word at every position).

### The Training Loop

Every epoch:
1. `optimizer.zero_grad()` — clear old gradients
2. `loss.backward()` — compute gradients via backpropagation (through time for RNNs)
3. `optimizer.step()` — update weights: param -= lr × gradient

### Vanishing Gradient Problem

Gradients shrink exponentially as they flow backward through time steps. With typical weight scale ~0.8:

- After 10 steps: gradient ≈ 0.1 (weak but usable)
- After 30 steps: gradient ≈ 0.001 (negligible)

This means RNNs can't effectively learn from long-range dependencies. Early words in a sentence get almost no gradient signal.

## Model Comparison

| Model | Context | Similarity | Parallelizable |
|-------|---------|-----------|----------------|
| N-gram | Fixed window (n-1 words) | None | N/A |
| Bengio | Fixed window (n-1 words) | Yes (embeddings) | Yes |
| RNN | Unlimited (theory), ~10-20 (practice) | Yes (embeddings) | No (sequential) |

## What Comes Next

Both Bengio and RNNs process context sequentially or through a fixed window. The Transformer uses **self-attention** to let every word look at every other word simultaneously — solving both the long-range dependency problem and the parallelism bottleneck.
