# Multi-head Latent Attention (MLA)

**Model:** DeepSeek-V3-671B

MLA with LoRA-compressed KV projections and RoPE.

**Dimensions:** batch=1, seq_len=2048, emb_dim=7168, 128 heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| optimized | Optimized LoRA projections with pre-computed RoPE |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 4.4813 | 0.0726 | 263.02 | 1.00x |
| optimized | 4.4173 | 0.0555 | 250.85 | 1.01x |
