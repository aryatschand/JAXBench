# Ragged Dot (Grouped Matmul)

Mixtral-8x7B grouped matrix multiplication for MoE expert routing (8 groups, 8192x4096x14336).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Batched jnp.matmul via vmap over groups |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 1.3641 | 0.0055 | 705.26 | 1.00x |
| optimized | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
