# Notebook 01: Autoregressive Generation and the KV Cache

## Core Idea

Naive autoregressive generation re-processes the entire sequence from scratch at every step — massively wasteful. The KV cache stores previously computed Keys and Values (which never change under causal attention), reducing per-step work from processing the full sequence to processing just 1 token.

## Key Concepts

### The Problem: Redundant Computation

Without caching, generating T tokens from a prompt of length n:

```
Step 1: process n+1 tokens
Step 2: process n+2 tokens
Step 3: process n+3 tokens
...
Step T: process n+T tokens

Total = nT + T(T+1)/2
```

Example: n=512 prompt, T=256 generated → 163,968 tokens processed for just 256 outputs.

### Why KV Cache Works

Under causal attention, token j's Key and Value depend ONLY on itself and earlier tokens — never on future tokens. So once computed, K_j and V_j never change. No need to recompute them.

### Two Phases of Cached Generation

**Phase 1 — Prefill (one time):**
- Feed entire prompt (e.g., 32 tokens)
- Compute and store K,V for all prompt tokens in the cache
- Get first prediction

**Phase 2 — Decode (loops):**
- Feed ONLY the 1 new token each step
- Compute Q, K, V for just this token
- Concatenate new K, V to cache
- New token's Q attends to all cached K's
- Pick next token, repeat

```
Naive:  32 + 33 + 34 + ... + 132 = ~8,000 tokens processed
Cached: 32 + 1 + 1 + 1 + ... + 1 = 132 tokens processed
```

### `start_pos` — Why It Matters

During decode, the new token needs the correct positional embedding. Without `start_pos`, every new token would get position 0's embedding — wrong.

```
Prefill: start_pos=0, tokens=[0,1,...,31]  → positions 0-31
Step 1:  start_pos=32, tokens=[new_token]  → position 32
Step 2:  start_pos=33, tokens=[new_token]  → position 33
```

### Cache Shape

For d_model=64, n_heads=4, d_k=16, after processing 5 tokens:

```
Cache K: (batch=1, heads=4, tokens=5, d_k=16)
Cache V: (batch=1, heads=4, tokens=5, d_k=16)
```

Each layer has its own cache. On the next step:
```
New token K: (1, 4, 1, 16) → concatenate → Cache K: (1, 4, 6, 16)
```

### Compute Cost Breakdown (per layer, per step)

**Two costs in each attention layer:**

1. **QKV Projections** — each token multiplied by 3 weight matrices:
```
Cost per token = 3 × d_model²
Cost for N tokens = 3 × N × d_model²
```

2. **Attention** — each query dots with all keys:
```
One query × all keys = seq_len × d_model
All queries × all keys = seq_len × seq_len × d_model = seq_len² × d_model
```

**Naive (per generation step):**
```
n_layers × (3 × seq_len × d_model² + seq_len² × d_model)
             └── re-project ALL tokens ──┘  └── ALL queries × ALL keys ──┘
```
seq_len grows each step → cost grows quadratically.

**Cached — Prefill (one time):**
```
n_layers × (3 × n_prompt × d_model² + n_prompt² × d_model)
```
Same formula, runs once on the prompt. K,V stored in cache.

**Cached — Decode (per step):**
```
n_layers × (3 × d_model² + seq_len × d_model)
             └── project 1 token ──┘  └── 1 query × all cached keys ──┘
```
The savings:
- Projections: 1 token instead of seq_len tokens → seq_len × cheaper
- Attention: 1 query instead of seq_len queries → seq_len × cheaper

**Example at seq_len=1000, d_model=4096:**
```
Naive per step:   3 × 1000 × 4096² + 1000² × 4096 ≈ 50B + 4B = 54B ops
Cached per step:  3 × 4096²        + 1000 × 4096   ≈ 50M + 4M = 54M ops
                                                       ~1000× fewer ops!
```

### Memory Cost (the tradeoff)

KV cache saves compute but costs memory:

```
KV cache memory = 2 × n_layers × seq_len × n_heads × d_k × bytes_per_element

For a 32-layer, d=4096 model in FP16:
  2048 tokens: ~1 GB per user
  4096 tokens: ~2 GB per user
  8192 tokens: ~4 GB per user
```

This is why inference is **memory-bound, not compute-bound** for LLMs — the KV cache for many concurrent users fills GPU memory before compute becomes the bottleneck.

### Speedup

| Metric | Without KV Cache | With KV Cache |
|--------|-----------------|---------------|
| Tokens processed per step | Entire sequence | 1 token |
| Computation scaling | O(T²) total | O(T) total |
| Memory overhead | Minimal | O(L × T × d) |

Speedup increases with longer generation — more steps = more wasted recomputation avoided.

### Interview Explanation

"The KV cache exploits the fact that in causal attention, past tokens' Keys and Values never change — they only depend on themselves and earlier tokens, not the future. So we compute them once and cache them. Each new token only needs to compute its own Q, K, V, append K and V to the cache, then attend to all cached keys. This turns per-step cost from O(sequence_length) to O(1) for projections, with the tradeoff being GPU memory proportional to sequence length."

## What Comes Next

Sampling strategies — how to choose the next token from the probability distribution (temperature, top-k, top-p) to produce coherent and diverse text.
