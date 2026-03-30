# Sparse Mixture of Experts

**Model:** Mixtral-8x7B

Sparse MoE with top-2 routing, 8 experts.

**Dimensions:** batch=1, seq_len=2048, emb_dim=4096, mlp_dim=14336

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 8.2921 | 0.0103 | 1.00x |
