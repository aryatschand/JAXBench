"""Sparse (Splash) Attention — Llama-3.1-70B GQA with causal mask.

Pallas TPU kernel implementation (fused attention).
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_sparse_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'sparse_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
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


def attention_kernel(q_ref, k_ref, v_ref, o_ref):
    h = pl.program_id(axis=0)
    i = pl.program_id(axis=1)

    q = q_ref[0, 0, :]              # (D,)
    k = k_ref[0, :, :]              # (S, D)
    v = v_ref[0, :, :]              # (S, D)

    # scores: (S,)
    scores = jnp.dot(k, q)          # (S,)

    # causal mask
    idx = jnp.arange(scores.shape[0])
    scores = jnp.where(idx <= i, scores, -1e30)

    # softmax
    m = jnp.max(scores)
    exp = jnp.exp(scores - m)
    denom = jnp.sum(exp)
    probs = exp / denom

    # output
    out = jnp.dot(probs, v)         # (D,)

    o_ref[0, 0, :] = out


def workload(q, k, v):
    S = CONFIG['seq_len']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    num_q_per_kv = H_q // H_kv

    k = jnp.repeat(k, num_q_per_kv, axis=0)
    v = jnp.repeat(v, num_q_per_kv, axis=0)

    out = pl.pallas_call(
        attention_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(H_q, S),
            in_specs=[
                pl.BlockSpec((1, 1, CONFIG['head_dim']), lambda h, i: (h, i, 0)),
                pl.BlockSpec((1, S, CONFIG['head_dim']), lambda h, i: (h, 0, 0)),
                pl.BlockSpec((1, S, CONFIG['head_dim']), lambda h, i: (h, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, CONFIG['head_dim']), lambda h, i: (h, i, 0)),
        ),
    )(q, k, v)

    return out


def benchmark(num_warmup=5, num_iters=100):
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
