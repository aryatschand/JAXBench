# Paged Attention

**Model:** Llama-3.1-70B

Paged KV-cache attention for inference decode with GQA.

**Dimensions:** num_seqs=16, max_seq_len=2048, 64 query heads, 8 KV heads, page_size=16

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` for the attention step |
| pallas | `jax.experimental.pallas.ops.tpu.paged_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | *pending* | — | — | — |
| optimized | *pending* | — | — | — |
| pallas | 1.6 | — | — | — |
