# Dense GEMM

Dense bfloat16 matrix multiplication at Llama-3.1-70B hidden-to-FFN scale (8192x8192x28672).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Upstream Pallas kernel (`jax.experimental.pallas.ops.tpu.matmul`) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | 5.4369 | 0.0077 | 707.81 | 1.00x |
| pallas | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
