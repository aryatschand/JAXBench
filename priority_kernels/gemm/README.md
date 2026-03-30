# Dense GEMM

**Model:** Llama-3.1-70B

Dense bfloat16 matrix multiplication at Llama-3.1-70B hidden-to-FFN scale.

**Dimensions:** M=8192, K=8192, N=28672

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | `jax.experimental.pallas.ops.tpu.matmul` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 5.4879 | 0.0092 | 701.23 | 1.00x |
| pallas | 22.6375 | 0.0189 | 170.00 | 0.24x |
