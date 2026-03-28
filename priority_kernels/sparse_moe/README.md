# Sparse Mixture of Experts

Mixtral-8x7B sparse MoE with top-2 expert routing, 8 experts, SwiGLU expert MLPs.

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Batched vmap over all experts (avoids sequential per-expert computation) |
| pallas | Upstream Pallas kernel (`jax.experimental.pallas.ops.tpu.megablox.gmm`) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 8.3069 | 0.0086 | 173.74 | 1.00x |
| optimized | — | — | — | *pending TPU run* |
| pallas | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
