# paged_attention

**Model:** Llama-3.1-70B
**Operator:** paged_attention
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
| baseline | 1.9649 |
| pallas/optimized | 1.0212 |
