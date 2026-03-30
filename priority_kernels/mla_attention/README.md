# Multi-head Latent Attention

**Model:** DeepSeek-V3-671B

MLA with LoRA-compressed KV and RoPE.

**Dimensions:** batch=1, seq_len=2048, emb_dim=7168, 128 heads

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 4.4833 | 0.0770 | 1.00x |
