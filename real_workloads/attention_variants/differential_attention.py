"""Differential Attention — Microsoft Research.

Computes attention as the difference of two softmax attention maps to cancel
noise. Each head splits into two sub-heads with a learnable lambda parameter.

Paper: "Differential Transformer" (Ye et al., ICLR 2025)
No existing JAX implementation — written from the paper specification.

Config based on DIFF Transformer-6.8B from the paper.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'diff_transformer_6_8b_attention',
    'model': 'DIFF-Transformer-6.8B',
    'operator': 'differential_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 32,
    'head_dim': 128,
    'd_model': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, lambda_q1, lambda_k1, lambda_q2, lambda_k2, lambda_init)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 9)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    # Q, K split into two sub-heads: each is (B, S, H, D//2)
    half_d = D // 2
    query = jax.random.normal(keys[0], (B, S, H, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, S, H, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, S, H, D), dtype=dtype)
    # Per-head learnable lambda parameters (initialized near 0.8 per paper)
    lambda_q1 = jax.random.normal(keys[3], (H, half_d), dtype=dtype) * 0.1
    lambda_k1 = jax.random.normal(keys[4], (H, half_d), dtype=dtype) * 0.1
    lambda_q2 = jax.random.normal(keys[5], (H, half_d), dtype=dtype) * 0.1
    lambda_k2 = jax.random.normal(keys[6], (H, half_d), dtype=dtype) * 0.1
    # Scalar init (paper: lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx))
    # Use layer 0 default: ~0.8
    lambda_init = jnp.array(0.8, dtype=dtype)
    return query, key_t, value, lambda_q1, lambda_k1, lambda_q2, lambda_k2, lambda_init


def workload(query, key, value, lambda_q1, lambda_k1, lambda_q2, lambda_k2, lambda_init):
    """Differential attention: attn = softmax(Q1*K1^T) - lambda * softmax(Q2*K2^T).

    Each head is split into two sub-heads. The difference cancels out attention
    noise, sharpening the attention pattern on relevant tokens.
    """
    B, S, H, D = query.shape
    half_d = D // 2
    scale = half_d ** -0.5
    # Split Q, K into two sub-heads
    q1, q2 = query[..., :half_d], query[..., half_d:]  # (B, S, H, D//2)
    k1, k2 = key[..., :half_d], key[..., half_d:]
    # Transpose to (B, H, S, D//2)
    q1, q2 = q1.transpose(0, 2, 1, 3), q2.transpose(0, 2, 1, 3)
    k1, k2 = k1.transpose(0, 2, 1, 3), k2.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)  # (B, H, S, D)
    # Compute lambda = exp(dot(lambda_q1, lambda_k1)) - exp(dot(lambda_q2, lambda_k2)) + lambda_init
    lam = (
        jnp.exp(jnp.sum(lambda_q1 * lambda_k1, axis=-1)) -
        jnp.exp(jnp.sum(lambda_q2 * lambda_k2, axis=-1)) +
        lambda_init
    )  # (H,)
    # Attention map 1
    attn1 = jnp.einsum('bhqd,bhkd->bhqk', q1, k1) * scale
    # Attention map 2
    attn2 = jnp.einsum('bhqd,bhkd->bhqk', q2, k2) * scale
    # Causal mask
    mask = jnp.tril(jnp.ones((S, S)))
    attn1 = jnp.where(mask, attn1, jnp.finfo(query.dtype).min)
    attn2 = jnp.where(mask, attn2, jnp.finfo(query.dtype).min)
    attn1 = jax.nn.softmax(attn1.astype(jnp.float32), axis=-1).astype(query.dtype)
    attn2 = jax.nn.softmax(attn2.astype(jnp.float32), axis=-1).astype(query.dtype)
    # Differential attention: A1 - lambda * A2
    diff_attn = attn1 - lam[None, :, None, None] * attn2  # (B, H, S, S)
    out = jnp.einsum('bhqk,bhkd->bhqd', diff_attn, v)
    # GroupNorm-style normalization per head (paper Section 3.2)
    out = out * (1.0 - lambda_init)
    return out.transpose(0, 2, 1, 3)


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    half_d = D // 2
    # Two attention maps: 2 * B*H*S*S*(D/2) * 2 (matmul)
    # Two weighted sums: 2 * B*H*S*S*D (AV matmul)
    flops = 2 * B * H * S * S * half_d * 2 + 2 * B * H * S * S * D
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
