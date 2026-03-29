# Megablox Grouped Matrix Multiply

**Model:** Qwen3-235B-A22B

Grouped matmul for MoE: each expert group gets its own weight matrix.

**Dimensions:** 128 experts, 8 experts/token, emb_dim=4096, mlp_dim=1536, seq_len=2048

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | `jax.experimental.pallas.ops.tpu.megablox.gmm` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | *pending* | — | — | — |
| pallas | 21.3 | — | — | — |
