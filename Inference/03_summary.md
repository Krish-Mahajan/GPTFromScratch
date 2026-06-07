# Notebook 03: Building a Complete Inference Engine

## Core Idea

Combine KV cache (fast generation) with sampling strategies (quality generation) into a complete inference pipeline. The key insight: prefill and decode are fundamentally different workloads requiring different optimization strategies.

## Key Concepts

### Prefill vs Decode — Two Phases

**Prefill** (process the prompt):
- All n prompt tokens processed in parallel, one forward pass
- Compute-bound — GPU doing lots of math on many tokens
- Weight matrices loaded once, reused across all n tokens (high arithmetic intensity)

**Decode** (generate tokens):
- One new token per step, sequential
- Memory-bound — GPU spends most time loading cached K,V vectors from memory
- Very little math per byte loaded (low arithmetic intensity)

Restaurant analogy: Prefill = kitchen prepares your dish (lots of parallel work). Decode = waiter delivers courses one at a time (sequential, mostly walking/carrying).

### Compute Cost per Layer

**Prefill (all n tokens at once):**
```
Projections: 3 × n × d_model²    ← Q,K,V for all n tokens
Attention:   n² × d_model         ← all queries × all keys
```
n = number of prompt tokens.

**Decode (1 token per step, at step t):**
```
Projections: 3 × d_model²         ← Q,K,V for 1 token only
Attention:   (n + t) × d_model    ← 1 query × all cached keys
```

The difference:
- Projections: n× less work (1 token vs n tokens)
- Attention: n× less work (1 query vs n queries)

### Arithmetic Intensity (FLOPs per byte loaded)

```
Prefill AI = O(n)   ← high: n tokens reuse same weight matrices → compute-bound
Decode AI  = O(1)   ← low: 1 token, load all KV cache just for one dot product → memory-bound
```

This is why:
- Prefill optimization → maximize GPU compute (tensor parallelism, batch prompts)
- Decode optimization → minimize memory access (quantized KV cache, maximize bandwidth)

Production systems often run prefill and decode on different hardware configurations.

### The GenerationEngine Class

Separates prefill and decode explicitly:

```python
# Phase 1: Prefill (one time)
logits, kv_caches = model(prompt_tokens, kv_caches=None, start_pos=0)
next_token = sample(logits[:, -1, :])

# Phase 2: Decode (loop)
for step in range(max_new_tokens):
    logits, kv_caches = model(next_token, kv_caches=kv_caches, start_pos=cur_pos)
    cur_pos += 1
    next_token = sample(logits[:, -1, :])
    if next_token == eos_token_id:
        break
```

### Streaming Generation

Yield tokens one at a time as they're generated — simulates the "typing" effect in ChatGPT. Hides per-token latency behind human reading speed.

```python
for token_id, info in engine.generate(prompt, stream=True):
    print(chr(token_id), end="", flush=True)
```

### Where Time Is Actually Spent

Benchmarking with different prompt lengths (generating 100 tokens each):

- **Decode dominates total time (80-95%+)** regardless of prompt length
- **Prefill throughput >> decode throughput** — prefill processes many tokens in parallel, decode processes 1 at a time
- **Prefill time grows with prompt length** (quadratic attention) but decode time stays roughly constant

This is the key takeaway: **almost all inference time is spent in decode** — one token at a time, memory-bound, waiting for KV cache data. That's why decode optimization matters most in production (quantized KV cache, speculative decoding, batching multiple users' decode steps together).

### Production Metrics

| Metric | What It Measures | Typical Values |
|--------|-----------------|----------------|
| TTFT (Time to First Token) | Prefill latency | 50-500ms |
| ITL (Inter-Token Latency) | Per-token decode time | 10-50ms |
| Throughput | Decode speed | 30-150 tok/s per user |
| KV Cache Memory | Memory per user | 0.5-4 GB |

### KV Cache Memory for Real Models

```
Memory = 2 × n_layers × seq_len × d_model × dtype_bytes × batch_size

GPT-2 (12 layers, d=768, FP16):
  seq=2K:  ~75 MB per user

LLaMA-7B (32 layers, d=4096, FP16):
  seq=2K:  ~1 GB per user
  seq=8K:  ~4 GB per user

LLaMA-70B (80 layers, d=8192, FP16):
  seq=2K:  ~5 GB per user
  seq=8K:  ~20 GB per user
```

This is why serving many concurrent users is the main challenge — KV cache memory per user × number of users fills GPU memory fast.

### Streaming Generation

Yields tokens one at a time using a generator — creates the "typing" effect:

```python
for token_id, info in engine.generate(prompt, stream=True):
    print(chr(token_id), end="", flush=True)
```

Measures **inter-token latency (ITL)** — time between consecutive tokens. P99 latency matters more than mean in production (users notice occasional stutters).

### Performance Dashboard

Generating different lengths from same prompt:

```
Prefill: constant (same prompt size every time)
Decode: scales linearly (2× tokens = 2× time)
Throughput: flat (steady tok/s regardless of output length)
```

**Latency formula for production:**
```
total_time ≈ prefill_time + (num_tokens × per_token_time)
TTFT = prefill_time only
```

### The yield/return Bug (Python lesson)

A function with `yield` anywhere in it becomes a generator — it can never `return` a value normally, even on a different code path. Fix: split streaming and batch generation into separate methods. The dispatcher function (`generate`) just calls the appropriate one based on `stream` flag.

## What Comes Next

LoRA fine-tuning — making the model better at specific tasks without retraining from scratch.
