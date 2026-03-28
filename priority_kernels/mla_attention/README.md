# Multi-head Latent Attention (MLA)

DeepSeek-V3-671B MLA with LoRA-compressed KV projections and RoPE.

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | jax.nn.dot_product_attention for core attention (keeps LoRA projections) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 4.4647 | 0.0894 | 264.00 | 1.00x |
| optimized | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
