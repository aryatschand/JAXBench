# Megablox GMM

**Model:** Qwen3-235B-A22B

Grouped matmul for MoE.

**Dimensions:** 128 experts, 8/token, 4096x1536

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Pallas megablox grouped matmul with autotuned tiling |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 187.0371 | 1.7742 | 1.00x |
| pallas | 2.8248 | 0.0081 | **66.2x** |
