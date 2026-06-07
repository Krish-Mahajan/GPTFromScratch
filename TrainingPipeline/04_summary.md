# Notebook 04: Learning Rate Scheduling and Gradient Clipping

## Core Idea

Two safety mechanisms that prevent training from exploding: (1) learning rate scheduling controls how fast the model updates, and (2) gradient clipping caps how much any single step can change the weights.

## Key Concepts

### Learning Rate

How big of a step you take when updating weights:

```
new_weight = old_weight - learning_rate × gradient
```

- **LR too large:** overshoot the minimum, loss oscillates or explodes
- **LR too small:** progress is painfully slow, may never converge
- **LR just right:** fast progress without instability

Concrete example:
```
weight = 0.5, gradient = 2.0

lr = 0.1:   new_weight = 0.5 - 0.1 × 2.0 = 0.3   (big jump)
lr = 0.001: new_weight = 0.5 - 0.001 × 2.0 = 0.498 (tiny nudge)
```

**Interview explanation:** "The learning rate controls how much we adjust the model's weights after each training step. Too high and training becomes unstable. Too low and training is too slow. In practice, we start small (warmup), ramp up to a peak, then gradually decrease (cosine decay) — fast early progress and precise fine-tuning at the end."

### Linear Warmup (first W steps)

```
lr_t = lr_max × (t / W)
```

Learning rate starts at 0, grows linearly to peak:
```
W=1000, lr_max=3e-4

Step 0:    lr = 0
Step 500:  lr = 1.5e-4
Step 1000: lr = 3e-4 (peak reached)
```

**Why?** Random weights at the start produce unreliable gradients. A large LR would amplify this noise and cause the loss to explode. Start small, ramp up as the model stabilizes.

Typical warmup: 5-10% of total training steps.

### Cosine Decay (after warmup)

```
progress = (t - W) / (T - W)
lr_t = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
```

Smoothly decreases from peak to minimum following a cosine curve:
```
lr_max=3e-4, lr_min=1e-5, W=1000, T=10000

Step 1000 (start):   progress=0   → cos(0)=1   → lr = 3e-4 (peak)
Step 5500 (middle):  progress=0.5 → cos(π/2)=0 → lr = 1.55e-4
Step 10000 (end):    progress=1   → cos(π)=-1  → lr = 1e-5 (minimum)
```

**Why cosine shape?**
- Stays near peak early (gets good learning done)
- Fast decay in the middle (most reduction happens here)
- Gentle landing at the end (fine-tunes without overshooting)

Smoother than step decay (which has abrupt drops) — converges better in practice.

### Max-Norm Gradient Clipping

If total gradient norm exceeds threshold c, scale all gradients down proportionally:

```
norm = √(g1² + g2² + g3² + ...)
if norm > c:
    scale = c / norm
    all gradients × scale
```

**Example:**
```
Gradients: [3.0, 4.0, 0.0]
Norm = √(9 + 16 + 0) = 5.0
Threshold c = 1.0

5.0 > 1.0, so scale = 1.0/5.0 = 0.2
Clipped: [0.6, 0.8, 0.0]  ← norm is now exactly 1.0
```

**Two key properties:**
1. **Direction preserved** — ratios stay the same (3:4:0 → 0.6:0.8:0). Model still moves the right direction, just a shorter step.
2. **Only activates on spikes** — normal training is unaffected. Only fires when something abnormal happens.

**Why needed?** Without clipping, one bad batch with 100× normal gradients can undo thousands of steps of progress in a single update.

Typical threshold: 1.0 (used by GPT, LLaMA, Mistral).

### The Driving Analogy (for interviews)

- **Warmup** = starting slowly on an unfamiliar road (don't know the curves yet)
- **Cosine decay** = slowing down as you approach your destination (need precision to park)
- **Gradient clipping** = a speed limiter (even on steep downhill, max speed is capped)
