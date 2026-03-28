# RMSNorm

Llama-3.1-70B Root Mean Square Layer Normalization (batch=1, seq_len=2048, emb_dim=8192).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
