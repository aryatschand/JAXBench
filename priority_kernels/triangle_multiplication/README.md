# Triangle Multiplicative Update

**Model:** AlphaFold2

Triangle multiplication (outgoing).

**Dimensions:** N=768, C=64

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 1.3130 | 0.0170 | 1.00x |
