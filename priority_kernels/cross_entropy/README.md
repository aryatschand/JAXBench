# Fused Cross-Entropy Loss

**Model:** Llama-3.1-8B

Fused linear + softmax cross-entropy.

**Dimensions:** tokens=4096, hidden=4096, vocab=128256

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 7.6978 | 0.0092 | 1.00x |
