# Sparse (Splash) Attention

**Model:** Llama-3.1-70B

Causal GQA attention baseline for splash attention optimization.

**Dimensions:** batch=1, seq_len=2048, 64 query heads, 8 KV heads, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` |
| pallas | `jax.experimental.pallas.ops.tpu.splash_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | *pending* | — | — | — |
| optimized | *pending* | — | — | — |
| pallas | 5.9 | — | — | — |
