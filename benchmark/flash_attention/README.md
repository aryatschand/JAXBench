# flash_attention

**Model:** Baseline-MHA
**Operator:** causal_mha
**Model:** model
**Operator:** operator

## Variants

| File | Description |
|------|-------------|
| baseline.py | Vanilla JAX implementation |
| pallas.py | Pallas TPU kernel with autotuned block sizes |

## Benchmark Results (TPU v6e-1)

| Variant | Time (ms) |
|---------|----------:|
| baseline | 1.4919 |
| pallas/optimized | 0.6212 |
