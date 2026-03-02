"""ALiBi Attention — BLOOM 7B1 (bigscience/bloom).

Attention with Linear Biases: replaces positional embeddings with a simple
per-head linear bias added to attention logits. No learned position parameters.

Source: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/bloom/modeling_flax_bloom.py
Paper: "Train Short, Test Long" (Press et al., 2022)
"""
import math
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'bloom_7b_alibi_attention',
    'model': 'BLOOM-7B1',
    'operator': 'alibi_causal_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 32,
    'head_dim': 128,
}


def _build_alibi_slopes(num_heads):
    """Compute per-head ALiBi slopes (geometric sequence).

    For power-of-2 heads: slope_m = 2^(-8m/n) for m in 1..n.
    For non-power-of-2: interleave extra slopes from doubled sequence.
    """
    closest_p2 = 2 ** math.floor(math.log2(num_heads))
    base = 2.0 ** (-(2.0 ** -(math.log2(closest_p2) - 3)))
    slopes = jnp.array([base ** i for i in range(1, closest_p2 + 1)], dtype=jnp.float32)
    if closest_p2 != num_heads:
        extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest_p2) - 3)))
        remaining = min(closest_p2, num_heads - closest_p2)
        extra = jnp.array([extra_base ** (2 * i + 1) for i in range(remaining)], dtype=jnp.float32)
        slopes = jnp.concatenate([slopes, extra])
    return slopes


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, S, H, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, H, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, H, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """BLOOM-style causal attention with ALiBi position bias."""
    B, S, H, D = query.shape
    scale = D ** -0.5
    q = query.transpose(0, 2, 1, 3)  # (B, H, S, D)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    # Scaled dot-product
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale  # (B, H, S, S)
    # ALiBi: slope * distance, broadcast over query dim
    slopes = _build_alibi_slopes(H)  # (H,)
    positions = jnp.arange(S, dtype=jnp.float32)
    alibi = slopes[:, None, None] * positions[None, None, :]  # (H, 1, S)
    attn = attn + alibi
    # Causal mask
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, jnp.finfo(query.dtype).min)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
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
    flops = B * H * S * S * D * 4
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
