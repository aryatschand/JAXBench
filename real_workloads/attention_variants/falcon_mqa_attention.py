"""Multi-Query Attention (MQA) — Falcon 7B (tiiuae/falcon-7b).

Single KV head shared across all query heads. Reduces KV cache by factor of
num_heads vs MHA. Precursor to GQA.

Source: https://huggingface.co/tiiuae/falcon-7b
Paper: "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'falcon_7b_mqa_attention',
    'model': 'Falcon-7B',
    'operator': 'multi_query_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 71,
    'num_kv_heads': 1,
    'head_dim': 64,
    'rotary_dim': 64,
    'rope_theta': 10000.0,
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    Hq, Hkv, D = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """MQA: single KV head broadcast to all query heads, with RoPE."""
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv
    # Apply RoPE
    cos, sin = _compute_rope(D, S, CONFIG['rope_theta'], query.dtype)
    query = _apply_rope(query, cos, sin)
    key = _apply_rope(key, cos, sin)
    # Expand single KV head to all query heads
    key = jnp.repeat(key, G, axis=2)      # (B, S, Hq, D)
    value = jnp.repeat(value, G, axis=2)
    q = query.transpose(0, 2, 1, 3)  # (B, Hq, S, D)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
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
    B, S, Hq, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_query_heads'], CONFIG['head_dim']
    flops = B * Hq * S * S * D * 4
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
