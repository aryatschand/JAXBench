# Ragged Paged Attention

Llama-3.1-70B variable-length paged attention for mixed prefill+decode (32 sequences, 2048 tokens).

## Variants

| Variant | Description |
|---------|-------------|
| baseline | Vanilla JAX implementation |
| pallas | Upstream Pallas kernel (`jax.experimental.pallas.ops.tpu.ragged_paged_attention`) |

## Benchmark Results (TPU v6e-1, JAX 0.6.2, bf16)

| Variant | Time (ms) | Std (ms) | TFLOPS | Speedup vs Baseline |
|---------|----------:|----------:|-------:|--------------------:|
| baseline | — | — | — | *pending TPU run* |
| pallas | — | — | — | *pending TPU run* |

*Results collected on Google Cloud TPU v6e-1 (single chip), JAX 0.6.2, bfloat16, median of 100 iterations with 5 warmup.*
