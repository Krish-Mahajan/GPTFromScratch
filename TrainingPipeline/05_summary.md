# Notebook 05: The Complete Training Loop

## Core Idea

Assemble all components from notebooks 01-04 (tokenizer, dataset, optimizer, LR scheduling, gradient clipping) into a single working training pipeline. Train a language model end-to-end and generate text.

## Key Concepts

### Cross-Entropy Loss

**What it measures:** How surprised the model is by the correct answer.

At each position, the model distributes confidence across all vocab options. Cross-entropy asks: "How much confidence did you put on the right answer?"

```
P(correct) = 0.9  → loss = -log(0.9)  = 0.1   (barely penalized)
P(correct) = 0.5  → loss = -log(0.5)  = 0.7   (moderate penalty)
P(correct) = 0.1  → loss = -log(0.1)  = 2.3   (harsh penalty)
P(correct) = 0.01 → loss = -log(0.01) = 4.6   (severe penalty)
```

**Why -log?** The log scale punishes low-confidence predictions harshly. A model that's 99% wrong should be punished far more than one that's 50% wrong. Linear penalty wouldn't create this urgency.

**For a full sequence** — average across all positions:
```
"The cat sat" (3 predictions):
  P("cat" | "The") = 0.4       → loss = 0.92
  P("sat" | "The cat") = 0.6   → loss = 0.51
  P("." | "The cat sat") = 0.2 → loss = 1.61

Average loss = (0.92 + 0.51 + 1.61) / 3 = 1.01
Perplexity = exp(1.01) = 2.75 → "choosing between ~3 options on average"
```

**In PyTorch:**
```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
#                      (B*T, 50257)               (B*T,)
# Flattens all positions, computes loss for each, averages into one number.
```

**Interview explanation:** "Cross-entropy loss is the negative log of the probability the model assigned to the correct token, averaged over all positions. The log scale ensures the model is heavily penalized for being confidently wrong — assigning near-zero probability to the correct answer is punished far more than being mildly uncertain."

### The Training Loop Cycle

Every iteration follows this pattern:

```
1. LOAD BATCH      — get (input, target) pairs from DataLoader
2. FORWARD PASS    — model predicts next tokens → logits (B, T, V)
3. COMPUTE LOSS    — cross-entropy between predictions and targets
4. BACKWARD PASS   — compute gradients for all weights
5. CLIP GRADIENTS  — cap gradient norm at 1.0
6. UPDATE LR       — warmup + cosine decay schedule
7. OPTIMIZER STEP  — AdamW updates weights
8. LOG METRICS     — record loss, perplexity, grad norm
```

This is the same loop used by GPT, LLaMA, Mistral — only the scale differs.

### The Complete Pipeline (all notebooks combined)

```
Raw Text  →  BPE Tokenizer (NB01)  →  token IDs
          →  Dataset + DataLoader (NB02)  →  batched (input, target) pairs
          →  AdamW Optimizer (NB03)  →  weight updates
          →  LR Schedule + Grad Clip (NB04)  →  stable training
          →  Training Loop (NB05)  →  trained model → text generation
```

### What to Watch During Training

- **Train loss dropping** → model is learning
- **Val loss dropping** → model is generalizing (not just memorizing)
- **Val loss plateaus while train keeps dropping** → overfitting
- **Loss spikes** → gradient explosion (clipping should handle this)
- **Perplexity** → intuitive measure: "choosing between X options per token"

### Training Infrastructure

**Evaluation function:**
- Switch to `model.eval()` (disables dropout)
- Run through all val batches, average loss
- Compute perplexity = exp(avg_loss)
- Switch back to `model.train()`

**Generate function (autoregressive):**
- Encode prompt → token IDs
- Loop: forward pass → take last position logits → temperature → sample → append
- Decode final sequence back to text

## What Comes Next

Scale up: BPE tokenization on larger datasets, distributed training across multiple GPUs, mixed precision (fp16/bf16), larger models with billions of parameters.
