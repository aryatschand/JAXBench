# Fused Cross-Entropy Loss

Llama-3.1-8B fused linear projection + softmax cross-entropy loss (4096 hidden, 128K vocab).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 7.6535 | 0.0088 | 562.51 | 1.00x |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
