# megablox_gmm

**Model:** Qwen3-235B-A22B
**Operator:** grouped_matmul
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
| baseline | 187.0371 |
| pallas/optimized | 2.8248 |
