# Grouped Query Attention

**Model:** Llama-3.1-405B

GQA with 128 query heads, 8 KV heads.

**Dimensions:** batch=1, seq_len=2048, emb_dim=16384

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 3.2404 | 0.0060 | 1.00x |
