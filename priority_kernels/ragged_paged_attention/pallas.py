"""Ragged Paged Attention (Pallas) — Llama-3.1-70B using upstream Pallas kernel.

Uses jax.experimental.pallas.ops.tpu.ragged_paged_attention for TPU-optimized
variable-length paged attention with async DMA and online softmax.
"""
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
    ragged_paged_attention,
)

CONFIG = {
    'name': 'ragged_paged_attention_llama70b_pallas',
    'model': 'Llama-3.1-70B',
    'operator': 'ragged_paged_attention',
    'max_num_batched_tokens': 2048,
    'max_num_seqs': 32,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    max_tokens = CONFIG['max_num_batched_tokens']
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )
    tokens_per_seq = max_tokens // max_seqs
    kv_len_per_seq = pages_per_seq * page_size
    kv_lens = jnp.full((max_seqs,), kv_len_per_seq, dtype=jnp.int32)
    page_indices = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(
        max_seqs, pages_per_seq
    )
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    """Pallas ragged paged attention kernel."""
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    return ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
        sm_scale=sm_scale,
        num_kv_pages_per_block=8,
        num_queries_per_block=64,
        vmem_limit_bytes=32 * 1024 * 1024,
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
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    tokens_per_seq = CONFIG['max_num_batched_tokens'] // max_seqs
    kv_len = CONFIG['pages_per_seq'] * CONFIG['page_size']
    flops = max_seqs * H_q * (4 * tokens_per_seq * kv_len * D)
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
