# Notebook 04: LoRA Fine-Tuning from Scratch

## Core Idea

Full fine-tuning updates all parameters (needs 112 GB for a 7B model). LoRA freezes original weights and adds tiny trainable matrices that capture task-specific changes. The key insight: fine-tuning weight changes are low-rank — they can be compressed ~128× without quality loss.

## Key Concepts

### Why Fine-Tuning Updates Are Low-Rank

When you fine-tune W (4096×4096 = 16M params), the change ΔW = W_finetuned - W_pretrained has most of its "energy" in just a few singular values:

```
σ₁ = 12.5    ← most of the change
σ₂ = 8.3
σ₃ = 3.1
σ₄ = 1.2
...
σ₁₆ = 0.01   ← negligible from here
...
σ₄₀₉₆ = 0.0  ← zero
```

Effective rank is ~4-16. Fine-tuning doesn't change everything — it nudges the model in a specific, low-dimensional direction.

### The LoRA Formula

```
W = W₀ + (α/r) × B × A

W₀: (d_out × d_in) — frozen pretrained weights
B:  (d_out × r)    — initialized to ZEROS
A:  (r × d_in)     — initialized to random Gaussian
r:  rank (4, 8, or 16 typically)
α:  scaling factor (typically r or 2r)
```

**Forward pass — two paths:**
```
h = W₀x + (α/r) × B × A × x
    └ frozen ┘   └ trainable ┘
```

**Why B=0?** At initialization, ΔW = B×A = 0 — model behaves exactly like pretrained. Training gradually learns the update from there. If both were random, the model would be immediately perturbed.

### Parameter Savings

```
Full weight: d² params
LoRA:        2 × d × r params
Compression: d / 2r

Example: d=4096, r=16
  Full: 16,777,216 params
  LoRA: 131,072 params
  Compression: 128×
```

### Memory Savings (7B model)

```
Full Fine-Tuning:
  Model weights (FP16):     14 GB
  Gradients (FP32):         28 GB
  Optimizer states (FP32):  56 GB
  Total:                    98 GB ← needs multiple GPUs

LoRA Fine-Tuning:
  Model weights (FP16):     14 GB (frozen, loaded once)
  LoRA params + grads + opt: ~0.1 GB
  Total:                    ~14 GB ← fits on one GPU!
```

### Where to Apply LoRA

In a transformer's attention layer:
- Original paper: W_Q and W_V only (best results per parameter)
- Later work (QLoRA etc.): all four (W_Q, W_K, W_V, W_O)
- Not typically applied to FFN layers

**Why attention but not FFN?**
- Attention learns sparse, structured relationships (subject→verb, pronoun→noun). Fine-tuning redirects *which* tokens attend to *which* — a low-dimensional change. ΔW is genuinely low-rank.
- FFN stores factual knowledge and patterns densely across all dimensions. Fine-tuning changes spread broadly — ΔW has higher effective rank. LoRA with small r loses more info here.
- For a fixed parameter budget, attention gives more bang per parameter. With larger budgets/ranks, FFN can help too — it's a tradeoff, not a hard rule.

### The LoRALinear Class

**Init:**
- Takes an existing `nn.Linear` and wraps it
- Creates two small matrices A (r × d_in) and B (d_out × r)
- B=0 → model unchanged at start. A=small random → breaks symmetry
- Freezes original weights (no gradients flow to them)

**Forward pass — two parallel paths:**
```
x (batch, seq, 256)
  │
  ├── original(x) → (batch, seq, 256)           ← frozen, no gradients
  │
  └── F.linear(x, A) → (batch, seq, 8)          ← compress 256 → 8 (bottleneck)
      F.linear(..., B) → (batch, seq, 256)       ← expand 8 → 256
      × scaling (α/r)                            ← control magnitude
  │
  └── add them → (batch, seq, 256)
```

The LoRA path is a bottleneck — only the low-rank "essence" of the update passes through 8 dimensions.

```python
def forward(self, x):
    base_output = self.original(x)
    lora_output = F.linear(F.linear(x, self.A), self.B) * self.scaling
    return base_output + lora_output
```

### Merging (Zero Inference Overhead)

After training, bake LoRA permanently into original weight:
```python
def merge(self):
    self.original.weight += self.scaling * (self.B @ self.A)
    # B(256,8) × A(8,256) = (256,256) full-size ΔW
    return self.original
```

Now it's a single `nn.Linear` again — no extra computation at inference. The LoRA wrapper is thrown away.

**Why this matters:** Cheap training (only update small A, B) AND fast inference (merged back, zero overhead). You get both.

Now you have a single weight matrix — no LoRA overhead at inference. This is why LoRA is "free" at serving time.

### The Chef Analogy (for interviews)

"LoRA is like an expert chef learning Japanese cuisine. They don't re-learn everything — knife skills, temperature control, plating are already mastered. They just add a few new techniques. The 'updates' to their ability are low-dimensional: a small number of new skills layered on top of vast existing knowledge. LoRA does the same for neural networks — it freezes the pretrained knowledge and only trains a small, low-rank update."

### Interview Explanation

"LoRA exploits the fact that fine-tuning weight changes are low-rank. Instead of updating a d×d matrix (millions of params), we decompose the update into two small matrices B (d×r) and A (r×d) where r is typically 8 or 16. This gives ~100× fewer trainable parameters, dramatically reducing memory. After training, the LoRA matrices can be merged back into the base weights for zero inference overhead. It's the standard approach for adapting large models to specific tasks."

### Injecting LoRA into a Model

```python
def inject_lora(model, rank=8, alpha=16, target_modules=('W_q', 'W_v')):
    # 1. Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Walk through model, replace target Linear layers with LoRALinear
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                setattr(module, target, LoRALinear(original, rank, alpha))

    # Result: only A, B matrices are trainable (~0.5-2% of total params)
```

For a 4-layer transformer targeting W_q and W_v: 8 LoRA layers injected.

### Rank vs Performance Tradeoff

Higher rank = more capacity but diminishing returns:
```
Rank 1:  loss ≈ 6.2, params = 2K      ← underfitting
Rank 4:  loss ≈ 5.9, params = 8K
Rank 8:  loss ≈ 5.6, params = 16K     ← sweet spot
Rank 16: loss ≈ 5.5, params = 32K     ← barely better
Rank 32: loss ≈ 5.5, params = 64K     ← diminishing returns
```

### Common Pitfalls

- **Device mismatch:** LoRA params created on CPU if inject_lora is called after `.to(device)`. Fix: call `.to(device)` again after injecting.
- **Append inside loop:** When measuring final loss per rank, append outside the training loop (not inside — that gives 200 values per rank).

### Key Hyperparameters

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| Rank (r) | 4-32 | Higher = more capacity, more params |
| Alpha (α) | r to 2r | Scales the update magnitude |
| Target modules | Q,V or all | Which weight matrices to adapt |
| Learning rate | 1e-4 to 3e-4 | Usually higher than full FT |

### Training from Scratch vs Full Fine-Tuning vs LoRA

| Approach | Starting point | Params updated | Data needed | Compute cost |
|----------|---------------|----------------|-------------|-------------|
| From scratch | Random weights | All | Trillions of tokens | $1M+ |
| Full fine-tune | Pretrained weights | All | 10K-1M examples | $100-10K |
| LoRA | Pretrained weights | ~1% (A, B only) | 1K-100K examples | $1-100 |

**Analogy:**
- From scratch = teaching a baby to speak, read, then write legal documents
- Full fine-tuning = taking a fluent writer and teaching them legal style
- LoRA = giving that writer a one-page cheat sheet of legal terms

Full fine-tuning and training from scratch update the same number of parameters — the only difference is the starting point (pretrained vs random). Starting from pretrained means you need far less data and compute to converge.

## What Comes Next

Model alignment — making models not just capable, but aligned with human values (DPO, RLHF) and exploring scaling laws.
