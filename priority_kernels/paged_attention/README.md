# Paged Attention

**Model:** Llama-3.1-70B

Paged KV-cache decode attention.

**Dimensions:** batch=32, 64 query heads, 8 KV heads, 128 pages/seq

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Pallas paged attention with async DMA |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 1.9649 | 0.0050 | 1.00x |
| pallas | 1.0212 | 0.0055 | **1.9x** |
