# RetNet Retention

**Model:** RetNet-6.7B

Multi-scale retention with exponential decay.

**Dimensions:** batch=1, seq_len=2048, 16 heads, head_dim=256

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 0.5200 | 0.0146 | 1.00x |
