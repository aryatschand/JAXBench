# Dense GEMM

**Model:** Llama-3.1-70B

Dense bfloat16 matrix multiplication at Llama-3.1-70B hidden-to-FFN scale.

**Dimensions:** M=8192, K=8192, N=28672 (hidden_dim → mlp_dim projection)

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | `jax.experimental.pallas.ops.tpu.matmul` with block_shape=(512,512) |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 5.4369 | 0.0077 | 707.81 | 1.00x |
| pallas | 22.7 | — | — | 0.24x |
