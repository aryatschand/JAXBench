import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

CONFIG = {
    'name': 'retnet_6_7b_retention',
    'model': 'RetNet-6.7B',
    'operator': 'multi_scale_retention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 16,
    'head_dim': 256,
    'd_model': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    return query, key_t, value


def retention_kernel(q_ref, k_ref, v_ref, g_ref, o_ref):
    bh = pl.program_id(axis=0)
    qb = pl.program_id(axis=1)

    q = q_ref[:, :].astype(jnp.float32)          # (TQ, D)
    k = k_ref[:, :].astype(jnp.float32)          # (S, D)
    v = v_ref[:, :].astype(jnp.float32)          # (S, D)

    gamma = g_ref[0, 0].astype(jnp.float32)

    TQ, D = q.shape
    S = k.shape[0]

    q_idx = qb * TQ + jnp.arange(TQ, dtype=jnp.float32)[:, None]
    k_idx = jnp.arange(S, dtype=jnp.float32)[None, :]

    dist = q_idx - k_idx
    mask = (dist >= 0).astype(jnp.float32)
    decay = jnp.exp(jnp.log(gamma) * dist) * mask

    qk = jnp.matmul(q, k.T)  # (TQ, S)
    w = qk * decay

    norm = jnp.sum(jnp.abs(w), axis=1, keepdims=True)
    norm = jnp.maximum(norm, 1.0)

    out = jnp.matmul(w, v) / norm
    o_ref[:, :] = out.astype(o_ref.dtype)


def workload(query, key, value):
    B, H, S, D = query.shape
    BH = B * H

    q = query.reshape(BH, S, D)
    k = key.reshape(BH, S, D)
    v = value.reshape(BH, S, D)

    gammas = 1.0 - jnp.exp2(-5.0 - jnp.arange(H, dtype=jnp.float32))
    gammas = jnp.repeat(gammas, B).reshape(BH, 1)

    q = q
    k = k
    v = v
    g = gammas.reshape(BH, 1, 1)

    tile_q = 128
    grid = (BH, S // tile_q)

    out = pl.pallas_call(
        retention_kernel,
        out_shape=jax.ShapeDtypeStruct((BH, S, D), query.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((tile_q, D), lambda bh, qb: (bh, qb)),
                pl.BlockSpec((S, D), lambda bh, qb: (bh, 0)),
                pl.BlockSpec((S, D), lambda bh, qb: (bh, 0)),
                pl.BlockSpec((1, 1), lambda bh, qb: (bh, 0)),
            ],
            out_specs=pl.BlockSpec((tile_q, D), lambda bh, qb: (bh, qb)),
        ),
    )(q, k, v, g)

    return out.reshape(B, H, S, D)


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
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    flops = 2 * B * H * S * S * D * 2
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
