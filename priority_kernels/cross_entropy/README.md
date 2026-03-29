# Fused Cross-Entropy Loss

**Model:** Llama-3.1-8B

Fused linear projection + log-softmax + NLL loss.

**Dimensions:** batch_tokens=4096, hidden_dim=4096, vocab_size=128256

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 7.6535 | 0.0088 | 562.51 | 1.00x |
