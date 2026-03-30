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
| baseline | 2.8794 | 0.5926 | 95.46 | 1.00x |
| optimized | 21.4722 | 0.1341 | 12.80 | 0.13x |
| pallas | 11.6198 | 0.0146 | 23.66 | 0.25x |
