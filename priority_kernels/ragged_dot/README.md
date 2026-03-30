# Ragged Dot (Grouped Matmul)

**Model:** Mixtral-8x7B

Grouped matrix multiplication.

**Dimensions:** 8 groups, M=8192, K=4096, N=14336

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 1.3661 | 0.0062 | 1.00x |
