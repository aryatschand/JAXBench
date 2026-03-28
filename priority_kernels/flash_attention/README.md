# Flash Attention (Causal MHA)

**Model:** Baseline-MHA

Standard causal multi-head attention.

**Dimensions:** batch=1, seq_len=4096, num_heads=32, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` (dispatches to flash attention on TPU) |
| pallas | `jax.experimental.pallas.ops.tpu.flash_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 2.8291 | 0.5127 | 97.16 | 1.00x |
| optimized | *pending* | — | — | — |
| pallas | 6.2 | — | — | 0.46x |

> **Note:** Pallas result uses a different configuration (64 heads, seq=2048 vs baseline 32 heads, seq=4096). Direct speedup comparison requires running with matched configs.
