# Flex Attention

Llama-3.1-70B attention with relative position bias score modification (64 heads, seq_len=2048).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | jax.nn.dot_product_attention with bias tensor |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | — | — | — | *pending TPU run* |
| optimized | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
