"""Grouped Matrix Multiply (Megablox GMM) — Qwen3-235B-A22B MoE dimensions.

Pallas TPU kernel implementation.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'megablox_gmm_qwen3_235b',
    'model': 'Qwen3-235B-A22B',
    'operator': 'grouped_matmul',
    'num_experts': 128,
    'num_experts_per_tok': 8,
    'emb_dim': 4096,
    'moe_mlp_dim': 1536,
    'seq_len': 2048,
}

_skip_jit = True


def create_inputs(dtype=jnp.bfloat16):
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


def gmm_kernel(x_ref, w_ref, o_ref):
    x = x_ref[:, :]                 # (T, K)
    w = w_ref[0, :, :]              # (K, N)
    y = jnp.dot(x, w, preferred_element_type=jnp.float32)
    o_ref[:, :] = y


def workload(lhs, rhs, group_sizes):
    G = CONFIG['num_experts']
    M, K = lhs.shape
    N = rhs.shape[2]
    tokens_per_expert = M // G

    block_lhs = (tokens_per_expert, K)
    block_rhs = (1, K, N)
    block_out = (tokens_per_expert, N)

    grid = (G,)

    return pl.pallas_call(
        gmm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), lhs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block_lhs, lambda i: (i, 0)),
                pl.BlockSpec(block_rhs, lambda i: (i, 0, 0)),
            ],
            out_specs=pl.BlockSpec(block_out, lambda i: (i, 0)),
        ),
    )(lhs, rhs)


def benchmark(num_warmup=2, num_iters=10):
    import time
    inputs = create_inputs()
    for _ in range(num_warmup):
        out = workload(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = workload(*inputs)
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
