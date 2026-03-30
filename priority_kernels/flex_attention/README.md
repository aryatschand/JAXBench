# Flex Attention

**Model:** Llama-3.1-70B

Attention with relative position bias.

**Dimensions:** batch=1, seq_len=2048, 64 heads, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 2.9112 | 0.4845 | 1.00x |
