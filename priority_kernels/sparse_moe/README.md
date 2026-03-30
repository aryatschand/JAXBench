# Sparse Mixture of Experts

**Model:** Mixtral-8x7B

Sparse MoE with top-2 expert routing, 8 SwiGLU expert MLPs.

**Dimensions:** batch=1, seq_len=2048, emb_dim=4096, mlp_dim=14336

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Batched vmap over all experts simultaneously |
| pallas | `jax.experimental.pallas.ops.tpu.megablox.gmm` |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 8.2964 | 0.0104 | 173.96 | 1.00x |
| optimized | 14.8174 | 0.0107 | 389.58 | 0.56x |
| pallas | 180.3568 | 0.0306 | 8.00 | 0.05x |
