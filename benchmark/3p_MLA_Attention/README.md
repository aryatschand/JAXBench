# mla_attention

**Model:** DeepSeek-V3-671B
**Operator:** mla_attention
**Model:** model
**Operator:** operator

## Variants

| File | Description |
|------|-------------|
| baseline.py | Vanilla JAX implementation |
| optimized.py | JAX with Pallas library imports |

## Benchmark Results (TPU v6e-1)

| Variant | Time (ms) |
|---------|----------:|
| baseline | 4.4833 |
| pallas/optimized | 4.2760 |
