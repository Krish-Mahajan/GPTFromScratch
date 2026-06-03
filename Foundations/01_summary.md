# Notebook 01: N-gram Language Models

## Core Idea

Predict the next word by counting how often word patterns appeared in training data. No learning — just frequency lookup.

## Key Concepts

### The Chain Rule + Markov Assumption

A language model estimates P(word sequence). The chain rule decomposes this into conditional probabilities:

```
P(w1, w2, ..., wn) = P(w1) × P(w2|w1) × P(w3|w1,w2) × ...
```

The Markov assumption truncates history to make this tractable:
- Bigram: P(wi | wi-1) — only looks at 1 previous word
- Trigram: P(wi | wi-2, wi-1) — looks at 2 previous words

### Probability Estimation

```
P(word | context) = Count(context, word) / Count(context)
```

### Laplace Smoothing

Prevents zero probabilities for unseen n-grams by adding α to all counts:

```
P_smooth(word | context) = (Count(context, word) + α) / (Count(context) + α × |V|)
```

### Temperature (for generation)

Controls randomness when sampling the next word:
- Low (0.3): nearly deterministic, repetitive
- Normal (1.0): raw probabilities
- High (2.0): flattened distribution, more random/creative

### Perplexity

Measures how "surprised" the model is by unseen text. Lower = better.

```
Perplexity = exp(-1/N × Σ log P(wi | context))
```

A perplexity of k means the model is as uncertain as choosing uniformly among k words.

## Why N-grams Fail

1. **Sparsity**: With vocab size V, possible bigrams = V². A 50K vocab has 2.5 billion possible bigrams but observes ~0.02% of them. Most entries are zero.

2. **No similarity**: "cat" and "dog" are completely unrelated symbols. Learning "the cat sat" teaches nothing about "the dog sat".

## What Comes Next

Neural language models solve both problems with **embeddings** — dense vectors where similar words are nearby. Knowledge transfers automatically between related words, and the model generalizes beyond exact pattern matches.
