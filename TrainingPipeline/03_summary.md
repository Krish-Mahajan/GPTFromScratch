# Notebook 03: Optimization — SGD, Momentum, and Adam

## Core Idea

The optimizer decides how to use gradients to update weights. SGD follows gradients blindly, Momentum adds inertia to smooth out noise, Adam adapts the learning rate per-weight based on gradient history.

## Key Concepts

### What is an Optimizer?

The gradient tells you *which direction* to move. The learning rate tells you *how far*. The optimizer decides *how to use that information* — follow blindly, or be smarter?

**Interview explanation:** "An optimizer determines how model weights are updated given computed gradients. SGD follows the gradient directly. Momentum smooths out noisy gradients by accumulating past direction. Adam adapts the learning rate for each weight individually — weights with small, consistent gradients get larger updates; weights with large, noisy gradients get smaller updates. AdamW is the standard for LLMs because it converges fastest with minimal tuning."

### SGD (Stochastic Gradient Descent)

The simplest strategy — just step downhill:

```
new_weight = weight - lr × gradient
```

Problem: in a narrow zigzag valley, it bounces left-right instead of going forward. Slow convergence.

### SGD with Momentum

A rolling ball that builds up speed in consistent directions:

```
velocity = 0.9 × old_velocity + gradient
new_weight = weight - lr × velocity
```

- If gradient keeps pointing the same way → velocity grows → faster progress
- If gradient oscillates → velocity averages out → smoother path
- Tradeoff: can overshoot the target and oscillate around it

### Adam (Adaptive Moment Estimation)

A personal coach for each weight, tracking two things:

**First moment (m)** — running average of gradients (direction):
```
m = 0.9 × m + 0.1 × gradient
```

**Second moment (v)** — running average of squared gradients (magnitude/variability):
```
v = 0.999 × v + 0.001 × gradient²
```

**Bias correction** (early steps m and v are too small since they started at 0):
```
m_hat = m / (1 - 0.9^t)
v_hat = v / (1 - 0.999^t)
```

**Update:**
```
weight -= lr × m_hat / (√v_hat + ε)
```

The key insight: dividing by √v_hat means:
- Weight with large, variable gradients → large v_hat → smaller step (it's unsure)
- Weight with small, consistent gradients → small v_hat → larger step (it knows where to go)

Each weight gets its own effective learning rate, automatically.

### AdamW (Adam with Decoupled Weight Decay)

Same as Adam but applies weight decay directly to weights (not through gradients):

```
# Adam update (normal)
weight -= lr × m_hat / (√v_hat + ε)

# Then weight decay (separate, decoupled)
weight -= lr × weight_decay × weight
```

Why decouple? In standard Adam, weight decay interacts with the adaptive learning rate in unintended ways. Decoupling makes regularization strength independent of the optimizer state. Used by every modern LLM.

### Why Adam Wins for LLMs

A transformer has millions of weights. Some (attention heads) get large gradients. Some (specific embedding dimensions) barely move. A single learning rate can't be optimal for all of them. Adam gives each weight its own adaptive rate — that's why it converges so much faster.

### Numerical Example (one step of each)

```
theta = 0.5, gradient = 0.2, lr = 0.01

SGD:      theta = 0.5 - 0.01 × 0.2 = 0.498
Momentum: v = 0.9×0 + 0.2 = 0.2, theta = 0.5 - 0.01 × 0.2 = 0.498 (same first step)
Adam:     adapts lr per-weight → theta = 0.499001 (different effective step size)
```

After many steps, Momentum accelerates in consistent directions, and Adam adapts per-weight — both outpace SGD significantly.

## What Comes Next

Learning rate scheduling (warmup + cosine decay) and gradient clipping — the safety mechanisms that prevent training from diverging.
