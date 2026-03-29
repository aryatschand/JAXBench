# Ragged Paged Attention

**Model:** Llama-3.1-70B

Variable-length paged attention for mixed prefill+decode serving.

**Dimensions:** max_tokens=2048, max_seqs=32, 64 query heads, 8 KV heads, page_size=16

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | `jax.experimental.pallas.ops.tpu.ragged_paged_attention` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | *pending* | — | — | — |
| pallas | 1.6 | — | — | — |
