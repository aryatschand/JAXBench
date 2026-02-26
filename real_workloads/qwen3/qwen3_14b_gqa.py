"""Grouped Query Attention (GQA) — Qwen 3 14B. Based on AI-Hypercomputer/maxtext.

Qwen 3 14B uses 40 query heads with 8 KV heads (5:1 group ratio).
Larger embedding dimension (5120) compared to the 8B (4096).
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'qwen3_14b_gqa',
    'model': 'Qwen3-14B',
    'operator': 'gqa_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 40,
    'num_kv_heads': 8,
    'head_dim': 128,
    'emb_dim': 5120,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    C = CONFIG
    B, S = C['batch'], C['seq_len']
    query = jax.random.normal(k1, (B, S, C['num_query_heads'], C['head_dim']), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, C['num_kv_heads'], C['head_dim']), dtype=dtype)
    value = jax.random.normal(k3, (B, S, C['num_kv_heads'], C['head_dim']), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """GQA: expand KV heads to match query heads, then standard attention."""
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv
    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    q = query.transpose(0, 2, 1, 3)
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
    C = CONFIG
    B, S, Hq, D = C['batch'], C['seq_len'], C['num_query_heads'], C['head_dim']
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
