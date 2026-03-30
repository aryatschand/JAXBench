# Grouped Query Attention

**Model:** Llama-3.1-405B

GQA with 128 query heads, 8 KV heads.

**Dimensions:** batch=1, seq_len=2048, emb_dim=16384

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Pallas splash attention with autotuned block sizes (bq=1024, bkv=1024) |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 3.2404 | 0.0060 | 1.00x |
| optimized | 1.4380 | 0.0050 | **2.3x** |
