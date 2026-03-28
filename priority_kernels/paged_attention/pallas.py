"""Paged Attention (Pallas) — Llama-3.1-70B using upstream JAX Pallas paged attention.

Uses jax.experimental.pallas.ops.tpu.paged_attention for TPU-optimized
paged KV-cache attention with async DMA page fetching.
"""
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

CONFIG = {
    'name': 'llama3_70b_paged_attention_pallas',
    'model': 'Llama-3.1-70B',
    'operator': 'paged_attention',
    'batch': 32,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, k_pages, v_pages, lengths, page_indices).

    Uses the layout expected by the Pallas paged attention kernel.
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    total_pages = B * pages_per_seq

    q = jax.random.normal(k1, (B, H_q, D), dtype=dtype)
    k_pages = jax.random.normal(k2, (H_kv, total_pages, page_size, D), dtype=dtype) * 0.02
    v_pages = jax.random.normal(k3, (H_kv, total_pages, page_size, D), dtype=dtype) * 0.02
    kv_len = pages_per_seq * page_size
    lengths = jnp.full((B,), kv_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32).reshape(B, pages_per_seq)

    return q, k_pages, v_pages, lengths, page_indices


def workload(q, k_pages, v_pages, lengths, page_indices):
    """Pallas paged attention kernel."""
    return paged_attention(
        q, k_pages, v_pages, lengths, page_indices,
        pages_per_compute_block=CONFIG['pages_per_seq'],
    )


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
    B = CONFIG['batch']
    H_q = CONFIG['num_query_heads']
    D = CONFIG['head_dim']
    kv_len = CONFIG['pages_per_seq'] * CONFIG['page_size']
    flops = B * H_q * (4 * kv_len * D)
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
