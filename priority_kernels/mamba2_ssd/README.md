# Mamba-2 State Space Duality

**Model:** Mamba-2-2.7B

Selective state space model via the SSD linear attention dual form.

**Dimensions:** batch=1, seq_len=2048, num_heads=64, head_dim=64, d_state=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 1.8024 | 0.4420 | 38.13 | 1.00x |
