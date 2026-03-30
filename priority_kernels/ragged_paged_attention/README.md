# Ragged Paged Attention

**Model:** Llama-3.1-70B

Variable-length paged attention.

**Dimensions:** max_tokens=2048, max_seqs=32, 64 query heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Pallas ragged paged attention with async DMA (32 pages/block) |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 191.9868 | 1.3604 | 1.00x |
| pallas | 0.8649 | 0.0058 | **222.0x** |
