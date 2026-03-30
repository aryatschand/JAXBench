# Paged Attention

**Model:** Llama-3.1-70B

Paged KV-cache attention for inference decode with GQA.

**Dimensions:** num_seqs=16, max_seq_len=2048, 64 query heads, 8 KV heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` for decode attention step |
| pallas | `jax.experimental.pallas.ops.tpu.paged_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 1.0423 | 0.0512 | 1.03 | 1.00x |
| optimized | 1.5580 | 0.0079 | 0.69 | 0.67x |
| pallas | 0.9242 | 0.0052 | 2.32 | 1.13x |
