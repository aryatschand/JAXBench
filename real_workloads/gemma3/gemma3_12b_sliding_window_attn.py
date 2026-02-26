"""Sliding Window Attention — Gemma 3 12B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'gemma3_12b_sliding_window_attn',
    'model': 'Gemma-3-12B',
    'operator': 'sliding_window_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 16,
    'num_kv_heads': 8,
    'head_dim': 256,
    'emb_dim': 3584,
    'sliding_window': 4096,
    'attn_logits_soft_cap': 50.0,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    Hq, Hkv, D = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """Sliding window attention with QK norm and soft capping."""
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv
    window = CONFIG['sliding_window']
    cap = CONFIG['attn_logits_soft_cap']
    query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-6)
    key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)
    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    attn = cap * jnp.tanh(attn / cap)
    pos = jnp.arange(S)
    dist = pos[:, None] - pos[None, :]
    mask = (dist >= 0) & (dist < window)
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
    eff_seq = min(S, CONFIG['sliding_window'])
    flops = B * Hq * S * eff_seq * D * 4
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
