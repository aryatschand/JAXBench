# Megablox Grouped Matrix Multiply

Qwen3-235B-A22B grouped matmul for MoE (128 experts, 8 experts/token, 4096x1536).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Upstream Pallas kernel (`jax.experimental.pallas.ops.tpu.megablox.gmm`) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | — | — | — | *pending TPU run* |
| pallas | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
