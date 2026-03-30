# Splash Attention

**Model:** Llama-3.1-70B

Causal GQA attention (splash baseline).

**Dimensions:** batch=1, seq_len=2048, 64 query heads, 8 KV heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Pallas splash attention with autotuned block sizes (bq=2048, bkv=2048) |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 1.4998 | 0.2690 | 1.00x |
| pallas | 0.6649 | 0.0046 | **2.3x** |
