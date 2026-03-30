# Flash Attention

**Model:** Llama-3.1-70B

Causal multi-head attention.

**Dimensions:** batch=1, seq_len=2048, num_heads=64, head_dim=128

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Pallas flash attention with autotuned block sizes |

## Benchmark Results

*TPU v6e-1 (us-east5-a), JAX 0.6.2, bfloat16, 100 iterations, 5 warmup*

| Variant | Time (ms) | Std (ms) | vs Baseline |
|---------|----------:|----------:|------------:|
| baseline | 1.4919 | 0.2812 | 1.00x |
| pallas | 0.6212 | 0.0045 | **2.4x** |
