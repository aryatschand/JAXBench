# SwiGLU MLP

**Model:** Llama-3.1-70B

SwiGLU feed-forward: gate + up projection with SiLU, then down projection.

**Dimensions:** batch=1, seq_len=2048, emb_dim=8192, mlp_dim=28672

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results

*TPU v6e-1, JAX 0.6.2, bfloat16, 100 iterations with 5 warmup*

| Variant | Time (ms) | Std (ms) | TFLOPS | vs Baseline |
|---------|----------:|----------:|-------:|------------:|
| baseline | 4.0781 | 0.0082 | 707.74 | 1.00x |
