# Notebook 04: Building a Tiny Language Model (Mini-GPT)

## Core Idea

Assemble all components from previous notebooks (embeddings, attention, transformer blocks) into a complete GPT-style model. Train it on Shakespeare text with next-token prediction and generate coherent text.

## Key Concepts

### Character-Level Tokenizer

Each character is a token — simplest possible tokenizer:

```
chars = sorted(set(text))  → ['\\n', ' ', '!', ..., 'a', 'b', ..., 'z']
vocab_size = 65            → 65 unique characters

encode("Hello") → [20, 43, 50, 50, 53]   # string → list of integers
decode([20, 43, 50, 50, 53]) → "Hello"   # integers → string
```

Why character-level instead of word-level:
- Tiny vocab (65 vs 50,000+) — model is much smaller
- No unknown words — any text can be encoded
- Tradeoff: sequences are longer, model needs more context to "see" full words

GPT uses BPE (byte-pair encoding) which is a middle ground — subword tokens.

### Training Loss: Cross-Entropy

```
Loss = -1/T × Σ log P(correct_char_t | all chars before t)
```

At each position, the model predicts the next character. Loss measures how much probability it assigned to the right answer:
- High probability to correct char → small loss (good)
- Low probability to correct char → big loss (bad)

The training data IS the labels — every character is both input (for positions before it) and target (for the position before it). This is **self-supervised** — no human labeling needed.

### Perplexity = exp(loss)

**Intuition: how many options the model is confused between at each step.**

Like a coin flip:
- Fair coin → perplexity = 2 (choosing between 2 options)
- Fair die → perplexity = 6 (choosing between 6 options)

For our model:
- Untrained (random): perplexity = 65 (randomly guessing among 65 chars)
- After training: perplexity ≈ 5-10 (narrowed down to ~5-10 plausible chars per step)
- Perfect model: perplexity = 1 (always 100% sure)

**Why exp(loss)?**
```
Model assigns P=0.1 to correct char → loss = -log(0.1) = 2.3
  → perplexity = exp(2.3) = 10 → "choosing among ~10 options"

Model assigns P=0.5 to correct char → loss = -log(0.5) = 0.69
  → perplexity = exp(0.69) = 2 → "down to ~2 choices"
```

exp() converts log-scale loss back to "number of choices" — intuitive for humans.

**Interview explanation:** "Perplexity measures how surprised the model is. A perplexity of 10 means the model is as uncertain as choosing uniformly among 10 equally likely options at every step. Lower = better."

### Data Batching

```python
def get_batch(split):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))  # 64 random starts
    x = stack([data[i:i+128] for i in ix])     # (64, 128) — input sequences
    y = stack([data[i+1:i+129] for i in ix])   # (64, 128) — targets (shifted by 1)
```

Each batch gives 64 × 128 = 8,192 next-character prediction tasks.

### Hyperparameters

```
BATCH_SIZE = 64       # sequences per batch
BLOCK_SIZE = 128      # context window (max chars model can see)
D_MODEL = 128         # embedding dimension
NUM_HEADS = 4         # attention heads (d_k = 128/4 = 32)
NUM_LAYERS = 4        # transformer blocks stacked
DROPOUT = 0.1         # 10% dropout
LEARNING_RATE = 3e-4  # standard for Adam + Transformers
MAX_ITERS = 3000      # total training steps
```

### MiniGPT Model Architecture

**What's new vs notebook 03:**

| Feature | Notebook 03 | Notebook 04 (MiniGPT) |
|---------|------------|----------------------|
| Positional encoding | Fixed sin/cos | Learned (nn.Embedding) |
| QKV projection | 3 separate matrices | 1 combined (d_model → 3×d_model), then split |
| Norm placement | Post-norm | Pre-norm (GPT-2 style) |
| Weight tying | No | Embedding = output projection |
| Generate method | None | Autoregressive sampling |

**Weight tying:** The output projection and token embedding share the same matrix. If the hidden state looks like "H"'s embedding → score "H" highly. One matrix, two jobs — saves parameters and enforces symmetry.

**Weight initialization:** All weights initialized to N(0, 0.02) — small random values to keep activations stable when stacking 4 blocks. Biases initialized to zero.

**Pre-norm vs post-norm:**
```
Post-norm (notebook 03): norm(x + sublayer(x))     ← normalize after residual
Pre-norm (MiniGPT):      x + sublayer(norm(x))     ← normalize before sublayer
```
Pre-norm keeps the residual path clean — gradients flow through unchanged.

### Forward Pass

```
idx (64, 128) integers
  → token_emb → (64, 128, 128)      look up character vectors
  → + pos_emb → (64, 128, 128)      add learned position vectors
  → dropout → (64, 128, 128)
  → 4 transformer blocks → (64, 128, 128)
  → final LayerNorm → (64, 128, 128)
  → output projection → (64, 128, 65)   score for each of 65 chars at each position
  → cross_entropy vs targets → scalar loss
```

### Generate Method (Autoregressive)

Produces one character at a time in a loop:

```
for each new token:
  1. Crop sequence to last 128 chars (context window)
  2. Forward pass → (1, seq_len, 65)
  3. Take LAST position's logits → (1, 65)
  4. Divide by temperature (low=sharp, high=flat)
  5. Top-k filter: keep only k best scores, rest → -∞
  6. Softmax → probabilities
  7. Sample one character from distribution
  8. Append to sequence → loop again
```

**Temperature:** divides logits before softmax
- 0.5: sharper → more deterministic, repetitive
- 1.0: unchanged
- 2.0: flatter → more random, creative

**Top-k:** only the k most likely characters can be sampled. Prevents picking very unlikely characters.

**Why sample instead of argmax?** Argmax always picks the same highest character → boring repetition. Sampling gives variety while still favoring likely options.

### Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for iter_num in range(3000):
    xb, yb = get_batch('train')                          # random batch (64, 128)
    logits, loss = model(xb, yb)                         # forward pass
    optimizer.zero_grad(set_to_none=True)                # clear gradients
    loss.backward()                                       # compute gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip if too large
    optimizer.step()                                      # update weights
```

**AdamW vs Adam:** AdamW applies weight decay directly to weights (mathematically correct for Transformers). Standard choice for all modern LLMs.

**Gradient clipping:** If the total gradient norm exceeds 1.0, scale all gradients down proportionally. Prevents exploding gradients from occasional bad batches.
```
norm = sqrt(g1² + g2² + g3² + ...)
If norm > 1.0: scale all gradients by (1.0 / norm)
```

**Evaluation (every 300 iters):**
- Switch to `model.eval()` (disables dropout)
- Average loss over 100 random batches (stable estimate)
- Compute perplexity = exp(loss) for both train and val
- Switch back to `model.train()`

**What to watch during training:**
- Both losses should drop
- Val loss close to train loss → good generalization
- Val stops dropping but train keeps going → overfitting

**Expected progression:**
```
Iter 0:    loss ≈ 4.17, perplexity ≈ 65   (random — guessing among 65 chars)
Iter 3000: loss ≈ 1.5,  perplexity ≈ 5    (narrowed to ~5 plausible chars per step)
```
