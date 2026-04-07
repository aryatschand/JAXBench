# sparse_attention

**Model:** Llama-3.1-70B
**Operator:** sparse_attention
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
| baseline | 1.4998 |
| pallas/optimized | 0.6649 |
