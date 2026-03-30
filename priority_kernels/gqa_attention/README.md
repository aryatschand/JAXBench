# Grouped Query Attention (GQA)

**Model:** Llama-3.1-405B

GQA with 128 query heads and 8 KV heads.

**Dimensions:** batch=1, seq_len=2048, emb_dim=16384, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` with GQA head expansion + projections |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 3.2354 | 0.0065 | 84.96 | 1.00x |
| optimized | 7.3763 | 0.0345 | 354.02 | 0.44x |
