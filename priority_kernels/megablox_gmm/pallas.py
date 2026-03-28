"""Megablox GMM (Pallas) — Qwen3-235B-A22B using upstream Pallas GMM kernel.

Uses jax.experimental.pallas.ops.tpu.megablox.gmm for TPU-optimized
grouped matrix multiplication with tiled computation.
"""
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

CONFIG = {
    'name': 'megablox_gmm_qwen3_235b_pallas',
    'model': 'Qwen3-235B-A22B',
    'operator': 'grouped_matmul',
    'num_experts': 128,
    'num_experts_per_tok': 8,
    'emb_dim': 4096,
    'moe_mlp_dim': 1536,
    'seq_len': 2048,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (lhs, rhs, group_sizes) — same layout as baseline."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    G = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    limit = 1 / (M * K)
    lhs = jax.random.uniform(k1, (M, K), dtype=dtype, minval=-limit, maxval=limit)
    lhs = lhs.astype(jnp.bfloat16).astype(dtype)
    rhs = jax.random.uniform(k2, (G, K, N), dtype=dtype, minval=-limit, maxval=limit)
    rhs = rhs.astype(jnp.bfloat16).astype(dtype)
    tokens_per_expert = M // G
    group_sizes = jnp.full((G,), tokens_per_expert, dtype=jnp.int32)
    return lhs, rhs, group_sizes


def workload(lhs, rhs, group_sizes):
    """Pallas megablox grouped matmul."""
    return gmm(lhs, rhs, group_sizes)


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
    G = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    flops = 2 * M * K * N
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
