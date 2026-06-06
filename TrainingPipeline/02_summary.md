# Notebook 02: Dataset and DataLoader

## Core Idea

Convert a flat sequence of token IDs into structured, shuffled, batched input-target pairs that the model can train on. The Dataset implements a sliding window; the DataLoader handles batching, shuffling, and streaming to GPU.

## Key Concepts

### Sliding Window: Input-Target Pairs

Target = input shifted right by 1 token. Every position predicts the next token.

```
tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]
context_length = 4

Sample at idx=0: x=[0,1,2,3]  y=[1,2,3,4]
Sample at idx=1: x=[1,2,3,4]  y=[2,3,4,5]
Sample at idx=2: x=[2,3,4,5]  y=[3,4,5,6]

Total samples = len(tokens) - context_length
```

### TextDataset Class

```python
class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y
```

### DataLoader: Batching + Shuffling

```python
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
```

- **batch_size=4**: grabs 4 samples, stacks into (4, context_length) tensor
- **shuffle=True**: randomizes order each epoch (prevents memorizing sequence order)
- **drop_last=True**: drops incomplete final batch (keeps shapes consistent)

Output shape: `(batch_size, context_length)` for both x and y.

### Why Batching Matters

Single-sample gradients are extremely noisy. Batching averages the loss across multiple samples → smoother, more reliable gradients.

```
Gradient variance with batch size B:
  Var[gradient_B] = Var[gradient_1] / B

Batch 1:  high variance (noisy updates)
Batch 32: 32× less variance (smooth updates)
```

### Why Shuffling Matters

Without shuffling, the model sees the same batch sequence every epoch → can memorize order. With shuffling, every epoch has different batch composition → forces the model to generalize.

### Context Length and Memory

Self-attention scales quadratically with context length:
```
Attention Memory ∝ T²

T=128:   1×
T=256:   4×
T=512:   16×
T=1024:  64×
T=2048:  256×
```

Doubling context length → 4× the attention memory.

### Streaming Dataset (for large data)

Real LLM data is too large for RAM. Use `np.memmap` to access token files from disk lazily — same Dataset interface, but only loads pages from disk when accessed.

### Gradient Accumulation (for limited GPU memory)

Simulates large batch size using multiple small forward-backward passes:

```
Effective batch size = mini_batch_size × accumulation_steps

Steps:
1. Forward + backward on mini-batch (gradients accumulate)
2. Repeat accumulation_steps times
3. Then: optimizer.step() + optimizer.zero_grad()
4. Divide loss by accumulation_steps so gradients average correctly
```

Example: batch_size=8 × accumulation_steps=4 = effective batch of 32, using only memory for 8.

### Full Pipeline

```
Raw text → Tokenizer → token IDs → TextDataset(sliding window) → DataLoader(batch+shuffle)
                                                                      ↓
                                                              (batch_size, context_length)
                                                              tensors ready for GPU
```

## What Comes Next

Build the optimizers (SGD, Momentum, Adam) that use these batches to update model weights.
