# Ragged Dot (Grouped Matmul)

**Model:** Mixtral-8x7B

Grouped matrix multiplication for MoE expert routing.

**Dimensions:** 8 groups, M=8192, K=4096, N=14336

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Batched `jax.vmap(jnp.matmul)` over groups |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 1.3689 | 0.0059 | 702.79 | 1.00x |
| optimized | 1.3704 | 0.0055 | 702.04 | 1.00x |
