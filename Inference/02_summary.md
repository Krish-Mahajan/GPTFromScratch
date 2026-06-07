# Notebook 02: Sampling Strategies for Text Generation

## Core Idea

The model outputs a probability distribution over 50,000+ tokens. How you pick one determines whether you get boring/repetitive text (greedy) or incoherent gibberish (too random). Sampling strategies find the sweet spot.

## Key Concepts

### The Problem

```
Model outputs probabilities:
  "mat"     → 0.25
  "couch"   → 0.18
  "floor"   → 0.15
  "table"   → 0.12
  ...
  "quantum" → 0.000001
```

- **Greedy** (always pick highest): always says "mat" → repetitive and boring
- **Pure random**: might pick "quantum" → incoherent
- **Good sampling**: enough randomness for creativity, enough constraint for coherence

### Temperature

Divide logits by τ before softmax. One knob for the diversity-quality tradeoff.

```
P(v) = softmax(logits / τ)
```

- **τ < 1** (e.g., 0.3): sharpens distribution → more deterministic, less diverse
- **τ = 1**: no change (raw probabilities)
- **τ > 1** (e.g., 2.0): flattens distribution → more random, more diverse
- **τ → 0**: becomes greedy (100% on top token)
- **τ → ∞**: becomes uniform random (equal probability for everything)

### Top-k Sampling

Keep only the k tokens with highest probability, zero out the rest, re-normalize, sample.

```
k=1:  greedy (only the best token)
k=10: choose from 10 best options
k=50: choose from 50 best options
```

**Problem:** k is fixed regardless of model confidence. If the model is 95% sure of one token, k=10 still allows 9 unlikely tokens in. If the model is truly uncertain among 20 options, k=10 cuts off valid choices.

### Top-p (Nucleus) Sampling

Sort probabilities largest to smallest. Keep adding tokens until cumulative probability reaches p. Zero out the rest, re-normalize, sample.

```
Confident model: "mat"=0.92 → only 1 token needed to reach p=0.9
Uncertain model: top 8 tokens needed to reach p=0.9 → 8 tokens kept
```

**Key advantage: adapts automatically to model confidence.** When sure, restricts heavily. When uncertain, allows more options. This is why top-p dominates in production.

### In Practice: Temperature + Top-p Together

1. Apply temperature (controls overall sharpness)
2. Apply top-p (cuts off the dangerous tail)
3. Sample from filtered distribution

```python
scaled_logits = logits / temperature
# sort, compute cumulative probs, mask tokens beyond p threshold
# sample from remaining
```

Temperature = how spread out. Top-p = hard cutoff on the tail.

### Production Defaults

| System | Temperature | Top-p |
|--------|------------|-------|
| OpenAI API | 1.0 | 1.0 |
| Claude | ~0.7 | ~0.9 |
| LLaMA | 0.6 | 0.9 |

### Greedy vs Sampling: The Repetition Problem

Greedy decoding picks the same token in similar contexts → leads to repetitive loops in longer text. Sampling introduces variety — even if the model slightly prefers "mat", sometimes picking "couch" or "floor" produces more natural, diverse text.

### Interview Explanation

"Sampling strategies control how we choose the next token from the model's probability distribution. Temperature scales the distribution — low temperature makes it peaked (nearly greedy), high temperature flattens it (more random). Top-k keeps a fixed number of candidates, while top-p adapts dynamically based on how confident the model is — keeping fewer options when confident, more when uncertain. In production, temperature and top-p are used together: temperature for overall randomness control, top-p to prevent sampling from the extreme tail of unlikely tokens."

## What Comes Next

Combine KV cache (efficient generation) with sampling strategies into a complete inference engine and benchmark end-to-end.
