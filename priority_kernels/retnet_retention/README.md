# RetNet Multi-Scale Retention

RetNet-6.7B multi-scale retention with per-head exponential decay (alternative to softmax attention).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 0.5186 | 0.0047 | 132.51 | 1.00x |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
