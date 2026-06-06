# Notebook 01: Tokenization and Byte Pair Encoding (BPE)

## Core Idea

Convert raw text into numbers that neural networks can process. BPE finds the sweet spot between character-level (tiny vocab, long sequences) and word-level (huge vocab, unknown words).

## Key Concepts

### Three Tokenization Strategies

| Strategy | Vocab Size | "transformer" → | Problem |
|----------|-----------|-----------------|---------|
| Character | ~256 | 11 tokens (t,r,a,n,s,...) | Sequences too long |
| Word | ~170,000+ | 1 token | Can't handle unseen words |
| Subword (BPE) | ~50,000 | 1-3 tokens | Sweet spot ✓ |

### The BPE Algorithm

A greedy algorithm that starts from individual characters and repeatedly merges the most frequent adjacent pair:

1. Split all words into characters + end marker `_`
2. Count all adjacent pairs across the corpus (weighted by word frequency)
3. Merge the most frequent pair into a new token
4. Repeat from step 2 until vocab reaches desired size

### Building Blocks (with example)

**Corpus:** "low low low low low lowest lowest newer newer newer wider wider wider wider"

**Step 1 — Word frequencies as character tuples:**
```
('l', 'o', 'w', '_'):              freq=5
('l', 'o', 'w', 'e', 's', 't', '_'): freq=2
('n', 'e', 'w', 'e', 'r', '_'):   freq=3
('w', 'i', 'd', 'e', 'r', '_'):   freq=4
```

The `_` end marker tells BPE where word boundaries are — without it, "low" inside "lowest" and standalone "low" would be indistinguishable.

**Step 2 — Count adjacent pairs:**
```
('l', 'o'): 5+2 = 7   ← from "low"(5) + "lowest"(2)
('o', 'w'): 5+2 = 7
('e', 'r'): 3+4 = 7   ← from "newer"(3) + "wider"(4)
('r', '_'): 3+4 = 7
('w', '_'): 5
('w', 'e'): 2
...
```

**Step 3 — Merge most frequent pair:**
```
Best pair: ('l', 'o') with count 7

Before: ('l', 'o', 'w', '_')     → After: ('lo', 'w', '_')
Before: ('l', 'o', 'w', 'e', 's', 't', '_') → After: ('lo', 'w', 'e', 's', 't', '_')
```

Two characters became one token. Repeat — next iteration might merge ('lo', 'w') → 'low'.

### Encoding (applying learned merges to new text)

Apply merge rules **in order** to any new text:
```
"widest" → ['w', 'i', 'd', 'e', 's', 't', '_']
  → apply rule ('e', 'r') — no match
  → apply rule ('er', '_') — no match
  → apply rule ('w', 'i') — match! → ['wi', 'd', 'e', 's', 't', '_']
  → ...
```

BPE can handle **unseen words** by breaking them into known subword pieces.

### Embedding Memory Cost

```
Memory = vocab_size × d_model × 4 bytes (float32)

V=256 (char):    256 × 4096 × 4 =    4 MB
V=50,257 (GPT-2): 50,257 × 4096 × 4 = 786 MB
V=100,000:       100,000 × 4096 × 4 = 1,563 MB
```

Larger vocab = more memory for the embedding table. This is why vocab size is an engineering tradeoff — not just "bigger is better."

### GPT-2's Tokenizer (tiktoken)

- Vocab size: 50,257
- Common English: ~4 chars per token (efficient)
- Rare names/code: ~2 chars per token (less efficient)
- Uses the same BPE algorithm, just trained on a massive corpus

## What Comes Next

Take token sequences and build the dataset pipeline — creating input-target pairs for next-token prediction using a sliding window approach.
