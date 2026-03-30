# Dense GEMM

**Model:** Llama-3.1-70B

Dense bfloat16 matrix multiplication.

**Dimensions:** M=8192, K=8192, N=28672

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Pallas tiled matmul with autotuned block sizes (1024x2048, bk=1024) |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 5.4830 | 0.0125 | 1.00x |
| pallas | 5.6237 | 0.2648 | 0.97x |
