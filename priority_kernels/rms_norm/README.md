# RMSNorm

**Model:** Llama-3.1-70B

Root Mean Square Layer Normalization.

**Dimensions:** batch=1, seq_len=2048, emb_dim=8192

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 0.1776 | 0.0039 | 0.47 | 1.00x |
