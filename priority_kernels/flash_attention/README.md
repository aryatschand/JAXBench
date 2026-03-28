# Flash Attention

Causal multi-head attention (32 heads, seq_len=4096, head_dim=128). Standard dot-product attention baseline.

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | jax.nn.dot_product_attention (dispatches to flash attention on TPU) |
| pallas | Upstream Pallas kernel (`jax.experimental.pallas.ops.tpu.flash_attention`) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 2.8291 | 0.5127 | 97.16 | 1.00x |
| optimized | — | — | — | *pending TPU run* |
| pallas | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
