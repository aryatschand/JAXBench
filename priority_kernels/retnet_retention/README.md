# RetNet Multi-Scale Retention

**Model:** RetNet-6.7B

Multi-scale retention with per-head exponential decay (alternative to softmax attention).

**Dimensions:** batch=1, seq_len=2048, num_heads=16, head_dim=256, d_model=4096

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 0.5186 | 0.0047 | 132.51 | 1.00x |
