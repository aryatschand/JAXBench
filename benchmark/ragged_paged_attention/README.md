# ragged_paged_attention

**Model:** Llama-3.1-70B
**Operator:** ragged_paged_attention
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
| baseline | 191.9868 |
| pallas/optimized | 0.8649 |
