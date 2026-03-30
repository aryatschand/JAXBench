# Sparse (Splash) Attention

**Model:** Llama-3.1-70B

Causal GQA attention baseline for splash attention optimization.

**Dimensions:** batch=1, seq_len=2048, 64 query heads, 8 KV heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` with GQA |
| pallas | `jax.experimental.pallas.ops.tpu.splash_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 1.5021 | 0.2742 | 91.50 | 1.00x |
| optimized | 1.8448 | 0.3909 | 74.50 | 0.81x |
| pallas | 5.8533 | 0.0110 | 23.48 | 0.26x |
