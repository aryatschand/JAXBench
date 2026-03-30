# Mamba-2 SSD

**Model:** Mamba-2-2.7B

State Space Duality linear attention.

**Dimensions:** batch=1, seq_len=2048, 64 heads, head_dim=64

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 1.8199 | 0.4402 | 1.00x |
