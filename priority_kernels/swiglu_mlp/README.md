# SwiGLU MLP

Llama-3.1-70B SwiGLU feed-forward layer (gate + up projection with SiLU, then down projection).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 4.0743 | 0.0081 | 708.40 | 1.00x |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
