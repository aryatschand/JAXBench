import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'mamba2_2_7b_ssd',
    'model': 'Mamba-2-2.7B',
    'operator': 'state_space_duality',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 64,
    'd_state': 128,
    'd_model': 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def _ssd_kernel(q_ref, k_ref, v_ref, lc_ref, o_ref):
    q = q_ref[0, 0, :, :].astype(jnp.float32)
    k = k_ref[0, 0, :, :].astype(jnp.float32)
    v = v_ref[0, 0, :, :].astype(jnp.float32)
    log_cum = lc_ref[0, 0, :]

    S = q.shape[0]

    scores = jnp.matmul(q, k.T)

    diff = log_cum[:, None] - log_cum[None, :]
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    L = jnp.exp(jnp.where(causal, diff, -1e30))

    scores = scores * L

    denom = jnp.sum(scores, axis=-1, keepdims=True)
    denom = jnp.where(jnp.abs(denom) < 1e-6, 1.0, denom)
    scores = scores / jnp.maximum(jnp.abs(denom), 1.0)

    out = jnp.matmul(scores.astype(v.dtype), v)

    o_ref[0, 0, :, :] = out.astype(o_ref.dtype)


def workload(query, key, value, A_log):
    B, H, S, D = query.shape

    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    grid = (B * H,)

    def idx_map(i):
        b = i // H
        h = i % H
        return (b, h, 0, 0)

    def idx_map_lc(i):
        b = i // H
        h = i % H
        return (b, h, 0)

    out = pl.pallas_call(
        _ssd_kernel,
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, S, D), idx_map),
                pl.BlockSpec((1, 1, S, D), idx_map),
                pl.BlockSpec((1, 1, S, D), idx_map),
                pl.BlockSpec((1, 1, S), idx_map_lc),
            ],
            out_specs=pl.BlockSpec((1, 1, S, D), idx_map),
        ),
    )(query, key, value, log_a_cumsum)

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
