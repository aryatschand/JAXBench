"""Sparse Attention (Pallas) — Llama-3.1-70B GQA using upstream splash attention.

Uses jax.experimental.pallas.ops.tpu.splash_attention for TPU-optimized
causal GQA with structured sparsity support.
"""
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as sak,
    splash_attention_mask as mask_lib,
)

CONFIG = {
    'name': 'llama3_70b_sparse_attention_pallas',
    'model': 'Llama-3.1-70B',
    'operator': 'sparse_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}

# Build splash attention kernel at module level
_S = CONFIG['seq_len']
_H_q = CONFIG['num_query_heads']
_heads_per_group = _H_q // CONFIG['num_kv_heads']
_mask = mask_lib.CausalMask(shape=(_S, _S))
_multi_head_mask = mask_lib.MultiHeadMask([_mask] * _H_q)
_splash_kernel = sak.make_splash_mha_single_device(
    _multi_head_mask, head_shards=1, q_seq_shards=1,
)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, k, v) in [num_heads, seq_len, head_dim] layout."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    S = CONFIG['seq_len']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (H_q, S, D), dtype=dtype) * (D ** -0.5)
    k = jax.random.normal(k2, (H_kv, S, D), dtype=dtype) * 0.02
    v = jax.random.normal(k3, (H_kv, S, D), dtype=dtype) * 0.02
    return q, k, v


def workload(q, k, v):
    """Splash attention with causal mask and GQA head expansion."""
    k_expanded = jnp.repeat(k, _heads_per_group, axis=0)
    v_expanded = jnp.repeat(v, _heads_per_group, axis=0)
    return _splash_kernel(q, k_expanded, v_expanded)


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
    H_q = CONFIG['num_query_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    flops = 4 * H_q * S * S * D
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
