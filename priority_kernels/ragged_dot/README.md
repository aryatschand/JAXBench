# Ragged Dot (Grouped Matmul)

**Model:** Mixtral-8x7B

Grouped matrix multiplication for MoE expert routing.

**Dimensions:** 8 groups, M=8192, K=4096, N=14336

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Batched `jnp.matmul` via `jax.vmap` over groups |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 1.3641 | 0.0055 | 705.26 | 1.00x |
| optimized | *pending* | — | — | — |
