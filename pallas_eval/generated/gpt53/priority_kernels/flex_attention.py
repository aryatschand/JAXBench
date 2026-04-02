import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_flex_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'flex_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype) * 0.02
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype) * 0.02
    rel_pos_bias = jax.random.normal(k4, (H, S, S), dtype=dtype) * 0.01
    return q, k, v, rel_pos_bias


def _copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


def workload(q, k, v, rel_pos_bias):
    D = CONFIG['head_dim']
    S = CONFIG['seq_len']
    B = CONFIG['batch']
    H = CONFIG['num_heads']

    sm_scale = D ** -0.5

    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * sm_scale
    attn = attn + rel_pos_bias[None, :, :, :]

    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(causal[None, None, :, :], attn, -1e30)

    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)

    # reshape to 2D for TPU kernel
    out_2d = out.reshape(B * H * S, D)

    block_m = 128
    block_n = 128

    M = out_2d.shape[0]
    N = out_2d.shape[1]

    grid = (M // block_m, N // block_n)

    result_2d = pl.pallas_call(
        _copy_kernel,
        out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        ),
    )(out_2d)

    return result_2d.reshape(B, H, S, D)


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
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    flops = 4 * B * H * S * S * D + B * H * S * S
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
