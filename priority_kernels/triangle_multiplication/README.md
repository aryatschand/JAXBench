# Triangle Multiplicative Update

AlphaFold2 triangle multiplication (outgoing) with gated projections (N=768, C=64).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
