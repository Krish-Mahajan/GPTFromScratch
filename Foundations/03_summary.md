# Notebook 03: Self-Attention and the Transformer

## Core Idea

Instead of processing words sequentially (RNN) or through a fixed window (Bengio), let every word directly look at every other word and decide how much to pay attention. This is self-attention — the foundation of all modern LLMs.

## Key Concepts

### Self-Attention: The Library Analogy

Every word simultaneously plays three roles:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information do I carry?"

Each word compares its Query against all Keys to find relevant words, then takes a weighted combination of their Values.

### The Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### Forward Pass (step by step with shapes)

Example: batch=1, seq_len=5, d_model=8, d_k=4

**Init**: Three learned projection matrices W_q, W_k, W_v (each 8→4) that extract query, key, value aspects from raw embeddings.

**Step 1 — Project input into Q, K, V:**
```
x:  (1, 5, 8)   — 5 words, each 8-dim embedding
Q = x @ W_q → (1, 5, 4)   — what each word is looking for
K = x @ W_k → (1, 5, 4)   — what each word advertises
V = x @ W_v → (1, 5, 4)   — what information each word carries
```

**Step 2 — Compute attention scores:**
```
scores = Q @ K^T / √d_k
       = (1,5,4) @ (1,4,5) / 2.0 = (1, 5, 5)
```
scores[i,j] = "how relevant is word j to word i?" Scaled to prevent softmax saturation.

**Step 3 — Mask + Softmax:**
```
scores.masked_fill(mask, -∞)   — block future positions (causal)
weights = softmax(scores)      — (1, 5, 5), each row sums to 1
```
Row i is a probability distribution: how much attention word i pays to each other word. Masked positions get weight 0.

**Step 4 — Weighted combination of values:**
```
output = weights @ V = (1,5,5) @ (1,5,4) = (1, 5, 4)
```
Each word's output = weighted blend of all value vectors using its attention weights. e.g., if word 2 paid 80% attention to word 0: output[2] ≈ 0.8*V[0] + 0.2*V[1].

**Complete flow:**
```
x (1,5,8) → project → Q,K,V (1,5,4)
  → Q@K^T/√d_k → scores (1,5,5)
  → mask + softmax → weights (1,5,5)
  → weights@V → output (1,5,4)
```

### Why Scale by √d_k

Each dot product sums d_k multiplications. If inputs have unit variance:
- Variance of dot product = d_k
- Standard deviation = √d_k

So with d_k=64, raw scores land in range ±8; with d_k=512, range ±22. Without scaling, softmax saturates (outputs near one-hot), gradients vanish, and the model stops learning. Dividing by √d_k brings scores back to ±1 regardless of dimension.

### Causal Mask (for GPT-style models)

In a decoder/language model, each word can only attend to previous words (no peeking at the future). Implemented by setting future positions to -∞ before softmax, which makes their attention weight exactly 0.

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill(mask, float('-inf'))
```

### Multi-Head Attention

Run h parallel attention heads, each learning a different type of relationship:
- Head 1 might focus on nearby words (local context)
- Head 2 might connect syntactic pairs (article↔noun)
- Head 3 might resolve coreferences ("it" → "mat")

**The "one big projection then split" trick:**

Instead of 4 separate small projections (4 matrix multiplies), we do one big projection and reshape (1 matrix multiply + free reshape). Mathematically identical, but GPUs are faster at one large matmul.

Example with x = [0.5, 1.0, 0.3, 0.8], d_model=4, num_heads=2, d_k=2:

```
Approach A (separate):
  x @ W_q1 = [0.91, 1.12]   ← head 1 query (one matmul)
  x @ W_q2 = [2.11, 2.36]   ← head 2 query (another matmul)

Approach B (one big, then split):
  W_q_big = [W_q1 | W_q2]   ← stack side by side
  x @ W_q_big = [0.91, 1.12, 2.11, 2.36]   ← one matmul
  Split: [0.91, 1.12] → head 1,  [2.11, 2.36] → head 2   ← free reshape
```

**Init** (d_model=16, num_heads=4, d_k=4):
```
W_q, W_k, W_v: (16 → 16) — all heads packed in one projection
W_o: (16 → 16) — mixes concatenated head outputs at the end
```

**Forward pass** (concrete example: batch=1, seq_len=3, d_model=4, num_heads=2, d_k=2):

**Step 1 — Project (all heads at once):**
```
Q = x @ W_q → (1, 3, 4)  — 4 dims = head1's 2 dims + head2's 2 dims packed together
```

**Step 2 — Reshape to separate heads:**
```
(1, 3, 4) → view → (1, 3, 2, 2) → transpose → (1, 2, 3, 2)
                     split 4 into    swap so heads     = (batch, heads, words, d_k)
                     2 heads × 2d    become a "batch"

Head 1: [[0.1, 0.2],   ← word 0     Head 2: [[0.5, 0.6],   ← word 0
          [0.3, 0.4],   ← word 1              [0.7, 0.8],   ← word 1
          [0.9, 1.0]]   ← word 2              [1.1, 1.2]]   ← word 2
```

**Step 3 — Attention per head (same as single-head, runs in parallel):**
```
Each head: Q@K^T/√d_k → softmax → @V
  scores: (1, 2, 3, 3)   ← each head has a 3×3 attention map
  weights: (1, 2, 3, 3)  ← each row sums to 1
  output: (1, 2, 3, 2)   ← each head gives each word a 2-dim result
         = (batch, heads, words, d_k)
```

**Step 4 — Concatenate heads:**
```
(1, 2, 3, 2) → transpose → (1, 3, 2, 2) → view → (1, 3, 4)
  regroup by word, then flatten heads together

  Word 0: [0.3, 0.4, 0.1, 0.7]   ← head1 output + head2 output glued
  Word 1: [0.5, 0.6, 0.2, 0.8]
  Word 2: [0.8, 0.9, 0.4, 0.9]
```

**Step 5 — Output projection (W_o):**
```
(1, 3, 4) @ W_o → (1, 3, 4)
```
Mixes the heads' outputs together — without W_o, each head's contribution stays isolated in its own dimensions.

**Complete flow:**
```
x (1,3,4) → project (1,3,4) → reshape (1,2,3,2) → attention (1,2,3,2)
  → concat (1,3,4) → W_o (1,3,4)
```
Output = same shape as input. Each word now carries information from all other words, seen from multiple perspectives (heads).

### Positional Encoding

**The problem**: Attention is permutation-invariant — "cat sat" and "sat cat" produce the same attention scores. The model can't tell word order.

**The solution**: Add a position vector (same size as d_model) to each word's embedding:

```
final_input = word_embedding + PE[position]
```

**The formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Example** (position 3, d_model=4):
```
Dim 0: sin(3 / 10000^(0/4)) = sin(3/1)   = 0.14    ← fast wave
Dim 1: cos(3 / 10000^(0/4)) = cos(3/1)   = -0.99
Dim 2: sin(3 / 10000^(2/4)) = sin(3/100) = 0.03    ← slow wave
Dim 3: cos(3 / 10000^(2/4)) = cos(3/100) = 1.00
```

**Intuition — the binary clock analogy:**

Binary counts position using bits that flip at different rates:
```
Bit 0: 0 1 0 1 0 1 0 1 ...  ← flips every position
Bit 1: 0 0 1 1 0 0 1 1 ...  ← flips every 2 positions
Bit 2: 0 0 0 0 1 1 1 1 ...  ← flips every 4 positions
```
Each position has a unique binary code. PE does the same thing but with smooth sin/cos waves instead of hard flips:
- Low dims = fast oscillation (tells exact position, like seconds hand)
- High dims = slow oscillation (tells rough region, like hour hand)

Every position gets a unique combination. And because waves are periodic, the difference between positions 5→8 looks the same as 100→103 — the model learns "3 apart" as one reusable pattern.

**Why not just use integers [0, 1, 2, ...]?**
- Grow unbounded — position 500 would dominate the embedding
- Can't generalize to longer sequences than seen in training
- No built-in notion of relative distance

Sin/cos stays bounded [-1, 1], works at any length, and encodes relative distance for free.

### Transformer Block

One complete block has 4 components:

| Component | What it does |
|-----------|-------------|
| Multi-Head Attention | Words talk to each other (communication) |
| Feed-Forward Network | Each word processes individually (computation) |
| LayerNorm (×2) | Normalizes values to prevent explosion/shrinkage |
| Dropout (×2) | Randomly zeros values during training (prevents overfitting) |

**Forward pass:**
```python
x = LayerNorm1(x + Dropout(Attention(x)))   # communicate → normalize
x = LayerNorm2(x + Dropout(FFN(x)))          # think → normalize
```

Each sub-layer wrapped with a residual connection. Stack N of these blocks = a Transformer.

### Residual Connections

```
output = x + sublayer(x)
```

The output is the **original input plus** what the sub-layer computed. Why:
- Gradients flow directly through the `+ x` path without degradation
- Deep stacks (12, 24, 96 layers) can train without vanishing gradients
- The sub-layer only needs to learn the "delta" — what to add, not rebuild from scratch

Also why FFN and attention must output the same shape as their input — addition requires matching shapes.

### LayerNorm

Normalizes each word's vector to mean=0, std=1, then applies learned scale (γ) and shift (β).

**Example:** word vector [4.0, 2.0, 0.0, 2.0]
```
mean = 2.0, std = 1.41
normalized = [(4-2)/1.41, (2-2)/1.41, (0-2)/1.41, (2-2)/1.41]
           = [1.41, 0.0, -1.41, 0.0]
output = γ * normalized + β   (γ, β are learned)
```

Normalizes across dimensions *within one word* — independent of batch size and other positions.

**Why LayerNorm instead of BatchNorm?**
- BatchNorm normalizes across the batch (needs multiple examples) — breaks with variable sequence lengths, padding, and batch=1 at inference
- LayerNorm normalizes within a single word — works identically whether batch=1 or batch=1000

### Dropout

During training, randomly zeros values with probability p (e.g., 0.1 = 10% killed):
```
Input:   [0.5, 1.2, 0.8, 0.3, 1.0, 0.7]
Training: [0.5, 0.0, 0.8, 0.0, 1.0, 0.7]  ← random values zeroed, survivors scaled up
Inference: [0.5, 1.2, 0.8, 0.3, 1.0, 0.7]  ← everything passes through unchanged
```

Forces redundancy — neurons can't rely on specific partners, so the model generalizes better. Different random mask each training step.

### Feed-Forward Network (FFN)

A 2-layer MLP applied independently to each word position:
```
x (1,6,16) → Linear1 (16→64) → GELU → Linear2 (64→16) → output (1,6,16)
```

- **Expand 4×** then **compress back** — gives a bigger "workspace" for complex transformations
- **GELU activation** — non-linearity between layers; without it, two linear layers collapse into one matrix
- **Independent per word** — word 0's FFN computation doesn't see word 1 (that's attention's job)

Think of it as:
- Attention = group discussion (words share information)
- FFN = individual thinking (each word processes what it heard)

### Parameter Count (d_model=32, num_heads=4, d_ff=128)

```
Attention:
  W_q: 32×32 = 1,024    ← weight matrix (no bias)
  W_k: 32×32 = 1,024
  W_v: 32×32 = 1,024
  W_o: 32×32 = 1,024
                 subtotal: 4,096

FFN:
  Linear1: 32×128 + 128(bias) = 4,224
  Linear2: 128×32 + 32(bias)  = 4,128
                 subtotal: 8,352

LayerNorm (×2):
  γ(32) + β(32) = 64 each × 2 = 128

Total per block: 4,096 + 8,352 + 128 = 12,576
```

Note: nn.Linear(in, out) has in×out weight params + out bias params.

### Complete Transformer Decoder (GPT-style)

**Pipeline:**
```
Input tokens → Token Embedding × √d_model → + Positional Encoding
  → Transformer Block × N
  → LayerNorm → Linear → logits over vocabulary
```

**What's learnable vs fixed:**

| Component | Learnable? | Why |
|-----------|-----------|-----|
| nn.Embedding | Yes | Table rows trained to capture word meaning |
| nn.Linear | Yes | Weight matrix + bias adjusted by gradient descent |
| LayerNorm (γ, β) | Yes | Scale and shift are tuned |
| PositionalEncoding | No | Fixed sin/cos formula |
| Dropout | No | Random mask, no parameters |
| Causal mask | No | Fixed upper triangle pattern |

**Where the parameters live** (vocab=20, d_model=32, heads=4, layers=3):

```
┌─────────────────────────────────────┬────────┬───────┐
│ Component                           │ Params │   %   │
├─────────────────────────────────────┼────────┼───────┤
│ FFN Linear layers (×3 blocks)       │ 25,056 │ 59.1% │
│ Attention W_q,W_k,W_v,W_o (×3)     │ 12,288 │ 29.0% │
│ Output Projection (32→20)           │    660 │  1.6% │
│ Token Embedding (20×32)             │    640 │  1.5% │
│ LayerNorm γ,β (7 total)             │    448 │  1.1% │
│                                     │        │       │
│ TOTAL                               │ 15,764 │  100% │
└─────────────────────────────────────┴────────┴───────┘
```

~90% of parameters live in FFN (59%) and Attention (29%). This ratio holds at any scale — GPT-3's 175B params follow the same distribution. The 4× expansion in FFN (d_model → 4×d_model → d_model) is why FFN dominates.

## Architecture Comparison

| Feature | N-gram | Neural LM | RNN | Transformer |
|---------|--------|-----------|-----|-------------|
| Context | n-1 words | n-1 words | Unlimited (theory) | Full sequence |
| Similarity | None | Embeddings | Embeddings | Embeddings |
| Long-range | None | Limited | ~10-200 tokens | Full sequence |
| Parallel | N/A | Yes | No | Yes |

## What Comes Next

Build a complete mini-GPT: train on real text with next-token prediction, generate coherent language, and measure perplexity.
