# Day 1 — Tensor Basics, Broadcasting, and Attention Shapes

## What I learned

- Creating PyTorch tensors and inspecting `shape`, `ndim`, and `dtype`
- Reshaping and reordering dimensions with `reshape`, `transpose`, and `permute`
- Adding and removing dimensions with `unsqueeze`
- Broadcasting rules for element-wise tensor operations
- Batched matrix multiplication
- How tensor shapes flow through a basic attention-score calculation

## Tensor shapes

Tensor shape describes the size of each dimension. A common Transformer input has shape:

```text
[B, T, D]
```

- `B` — batch size: independent samples processed together
- `T` — sequence length: number of tokens per sample
- `D` — embedding dimension: features representing each token

For example, `X` with shape `[2, 3, 4]` contains two samples, each with three tokens represented by four features.

## Shape operations

Given `x.shape == [2, 3, 4]`:

```python
x.reshape(2, 12)       # [2, 12]
x.transpose(1, 2)      # [2, 4, 3]
x.permute(0, 2, 1)     # [2, 4, 3]
x.unsqueeze(1)         # [2, 1, 3, 4]
```

`reshape` changes how the same number of elements is grouped. `transpose` swaps two dimensions, while `permute` specifies a complete dimension order. `unsqueeze` adds a dimension of size `1`.

## Broadcasting

Broadcasting allows compatible tensors to participate in an element-wise operation without manually copying values. Dimensions are compared from right to left; they are compatible when they are equal or when one of them is `1`.

```text
[2, 3, 4] + [4]       -> [2, 3, 4]
[2, 3, 4] + [1, 3, 1] -> [2, 3, 4]
```

The smaller tensor behaves as though its size-`1` dimensions were expanded. This is useful for applying shared offsets, masks, or parameters across batches and tokens.

## Matrix multiplication

For a batch of token embeddings and a projection matrix:

```text
X: [B, T, D]
W: [D, H]
X @ W: [B, T, H]
```

The final dimension of `X` (`D`) matches the first dimension of `W`. Each token embedding is projected from `D` features to `H` features, independently within every batch item.

## Attention-score shapes

With queries and keys:

```text
Q: [B, T, D]
K: [B, T, D]
K.transpose(-2, -1): [B, D, T]
```

The attention scores are calculated as:

```python
scores = Q @ K.transpose(-2, -1)
```

```text
scores: [B, T, T]
```

For each sample, the `[T, T]` matrix holds pairwise similarity scores: every query token is compared with every key token. Batch items remain independent; the batch dimension is used for efficient parallel computation rather than cross-sample attention.

## Files

- `tensor_playground.py` — small PyTorch experiments for tensor operations, broadcasting, matrix multiplication, and attention scores.

## Reflection

The key takeaway from Day 1 is that tracking tensor dimensions is essential for understanding and implementing attention. Once the meaning of each axis is clear, operations such as projections and `Q @ K.transpose(-2, -1)` become predictable.
