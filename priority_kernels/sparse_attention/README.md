# Sparse (Splash) Attention

Llama-3.1-70B causal GQA attention (64 query heads, 8 KV heads, seq_len=2048).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | jax.nn.dot_product_attention |
| pallas | Upstream Pallas kernel (`jax.experimental.pallas.ops.tpu.splash_attention`) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | — | — | — | *pending TPU run* |
| optimized | — | — | — | *pending TPU run* |
| pallas | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
