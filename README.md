# GPT From Scratch

Building GPT from scratch — educational notebooks progressing from n-grams to a complete language model with inference and fine-tuning.

## Structure

### Foundations
Build understanding from first principles:
1. **N-gram Language Models** — predicting next words by counting
2. **Neural Language Models** — word embeddings, Bengio's model, RNNs
3. **Self-Attention & Transformers** — attention mechanism, multi-head attention, positional encoding
4. **Building a Tiny Language Model** — complete mini-GPT trained on Shakespeare

### Training Pipeline
The engineering of training at scale:
1. **Tokenization & BPE** — byte pair encoding from scratch
2. **Dataset & DataLoader** — sliding window, batching, gradient accumulation
3. **Optimization** — SGD, Momentum, Adam, AdamW from scratch
4. **LR Scheduling & Gradient Clipping** — warmup, cosine decay, max-norm clipping
5. **Complete Training Loop** — end-to-end pipeline assembling all components

### Inference
Efficient generation and adaptation:
1. **Autoregressive Generation & KV Cache** — eliminating redundant computation
2. **Sampling Strategies** — temperature, top-k, top-p (nucleus) sampling
3. **Complete Inference Engine** — prefill/decode separation, streaming, benchmarking
4. **LoRA Fine-Tuning** — low-rank adaptation for parameter-efficient fine-tuning

## Hardware

Notebooks run on a P5.48xlarge instance (8x NVIDIA H100 80GB GPUs) via VS Code remote Jupyter kernel.

## Summary Documents

Each notebook has a corresponding `XX_summary.md` with key concepts, formulas, intuitive explanations, and interview-ready answers.
