# RMSNorm

**Model:** Llama-3.1-70B

Root Mean Square normalization.

**Dimensions:** batch=1, seq_len=2048, emb_dim=8192

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 0.1727 | 0.0060 | 1.00x |
