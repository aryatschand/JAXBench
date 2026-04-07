# gemm

**Model:** Llama-3.1-70B
**Operator:** dense_matmul
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
| baseline | 5.4830 |
| pallas/optimized | 5.6237 |
