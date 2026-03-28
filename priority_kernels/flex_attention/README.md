# Flex Attention

**Model:** Llama-3.1-70B

Attention with arbitrary score modification (relative position bias).

**Dimensions:** batch=1, seq_len=2048, num_heads=64, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | `jax.nn.dot_product_attention` with bias tensor |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | *pending* | — | — | — |
| optimized | *pending* | — | — | — |
