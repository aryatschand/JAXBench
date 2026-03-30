# Ragged Paged Attention

**Model:** Llama-3.1-70B

Variable-length paged attention for mixed prefill+decode.

**Dimensions:** max_tokens=2048, max_seqs=32, 64 query heads, 8 KV heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | `jax.experimental.pallas.ops.tpu.ragged_paged_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 200.5945 | 11.8335 | 0.69 | 1.00x |
| pallas | 1.5510 | 0.0055 | 88.62 | 129.33x |
