# Grouped Query Attention (GQA)

**Model:** Llama-3.1-405B

GQA with 128 query heads and 8 KV heads (16:1 ratio).

**Dimensions:** batch=1, seq_len=2048, emb_dim=16384, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` with GQA head expansion |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 3.2563 | 0.0074 | 84.42 | 1.00x |
| optimized | *pending* | — | — | — |
