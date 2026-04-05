# gqa_attention

**Model:** Llama-3.1-405B
**Operator:** gqa_attention
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
| baseline | 3.2404 |
| pallas/optimized | 1.4380 |
