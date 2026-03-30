# SwiGLU MLP

**Model:** Llama-3.1-70B

SwiGLU feed-forward layer.

**Dimensions:** batch=1, seq_len=2048, emb_dim=8192, mlp_dim=28672

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | Speedup |
|---------|----------:|----------:|--------:|
| baseline | 4.0748 | 0.0086 | 1.00x |
