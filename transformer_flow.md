# Lifecycle of a Token in GPT-2 (Andrej Walkthrough)

This doc is meant to help me (or anyone else) reason through how a single token flows through Andrej’s GPT-2 style code. 
It’s not trying to be perfect theory — just a step-by-step practical walkthrough.

---

## Input Setup

1. We assume we have a vocabulary of indexed tokens, something like:
   `{0: 'a', 1: 'alf', 2: 'alfa', ..., 100001: 'zebra'}`

2. Tokenize a query:
   `"How are you?" → ['How', ' are', 'you', '?']`

3. Encode tokens (map to vocab indices):
   `['How', ' are', 'you', '?'] → [5299, 553, 481, 30]`

4. During inference we might just have one sequence, but usually in training we have a batch. So inputs often look like:

   ```
   [
       [5299, 553, 481, 30],
       [4827, 817, 1033, 54924, 30],
       [50385, 11, 1412, 480, 413, 78922]
   ]
   ```

5. If the sequences aren’t the same length, some implementations pad them up to the max length:

   ```
   [
       [5299, 553, 481, 30, 0, 0],
       [4827, 817, 1033, 54924, 30, 0],
       [50385, 11, 1412, 480, 413, 78922]
   ]
   ```

   (Padding strategy varies — some models rely on `attention_mask` instead.)

---

## Embeddings

6. For each integer token ID, look up its row in the **token embedding matrix** `(vocab_size, n_embd)`.
   This gives a learned dense vector representation of the token.

7. For each position (0, 1, 2, …) look up its row in the **positional embedding matrix** `(block_size, n_embd)`.
   This gives the model a sense of *where* a token is in the sequence.

8. Add token embedding + positional embedding → now each token has both its meaning and its position baked into the vector.

---

## Transformer Block (the main loop)

The transformer is basically a stack of identical blocks. Each block is:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

### Step 1. LayerNorm

* Normalizes the embedding values.
* Why: keeps activations stable so training doesn’t blow up or vanish.

### Step 2. Multi-Head Self-Attention

9. We split the embedding into multiple “heads.”

   * Say `n_embd = 768`, `n_heads = 12` → each head has size 64.
   * Why: heads let the model look at different “aspects” of context in parallel.

10. Project the input embedding into 3 things: **Q (query), K (key), V (value)**.

* Technically this is done with a single linear layer that outputs `3 * n_embd`, then split.
* Shape: `(B, T, n_embd)` → `(B, n_heads, T, head_size)` for each Q/K/V.

11. Compute raw attention scores: `Q @ Kᵀ`.

* Shape: `(B, n_heads, T, T)`
* Each score says: *how much should token i pay attention to token j?*

12. Scale by `1/√head_size` (helps keep gradients sane).

13. Apply causal mask → make sure we only look at current and past tokens, never the future.

14. Softmax over the last dim → convert raw scores to probabilities (affinities).

15. Use these probs to take a weighted sum of values (V).

* Now each token is represented as a blend of the values of the tokens it attends to.(Including itself)

16. Concatenate all heads back together, then do a final projection to bring them back into `(B, T, n_embd)`.

17. Add the original input (residual connection).

* Why: makes optimization easier and keeps gradients flowing.

---

### Step 3. MLP (a.k.a. Feed-Forward Network)

18. First, project the embedding UP in dimensionality: `(n_embd → 4 * n_embd)`.

* Why: gives the model more “space” to mix and express features.

19. Apply a non-linearity (GeLU).

* Why: introduces non-linear decision boundaries, lets the model represent more complex functions.

20. Project back DOWN: `(4 * n_embd → n_embd)`.

* Why: squash it back to original embedding size, while keeping the richer representation.

21. Add residual connection (input + output again).

---

### Step 4. Repeat

* That whole attention + MLP block is repeated **n_layer times** (e.g. 12 for GPT-2 small).

---

## Final Steps

22. After the last block, apply a final LayerNorm.

23. Project into vocab size: `(B, T, vocab_size)`.

* Each position gets raw **logits** over the whole vocabulary.

24. Focus on the last token: `logits[:, -1, :]`.

25. Apply softmax → probability distribution over the next token.

26. Sample from this distribution (top-k, top-p, or greedy).

27. Append the chosen token to the sequence.

28. Loop until you hit max length or an `<|endoftext|>` token.

---

## TL;DR Quick Flow

1. Tokenize → vocab indices
2. Embeddings (token + position)
3. For each block:

   * Norm → Multi-Head Attention → Residual
   * Norm → MLP → Residual
4. Final Norm
5. Project to vocab
6. Softmax + sample next token
7. Repeat

---