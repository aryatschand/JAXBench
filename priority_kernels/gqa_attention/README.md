# Grouped Query Attention (GQA)

Llama-3.1-405B grouped-query attention with 128 query heads and 8 KV heads.

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | jax.nn.dot_product_attention with GQA head expansion |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 3.2563 | 0.0074 | 84.42 | 1.00x |
| optimized | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
