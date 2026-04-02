import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'flash_attention_baseline',
    'model': 'Baseline-MHA',
    'operator': 'causal_mha',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    value = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return query, key_t, value


def attention_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]

    S = q.shape[0]
    D = q.shape[1]
    scale = D ** -0.5

    attn = jnp.matmul(q, k.T) * scale
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)

    o_ref[...] = out


def workload(query, key, value):
    B, H, S, D = query.shape
    BH = B * H

    q = query.reshape(BH, S, D)
    k = key.reshape(BH, S, D)
    v = value.reshape(BH, S, D)

    block = (1, S, D)

    out = pl.pallas_call(
        attention_kernel,
        out_shape=jax.ShapeDtypeStruct((BH, S, D), query.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(BH,),
            in_specs=[
                pl.BlockSpec(block, lambda i: (i, 0, 0)),
                pl.BlockSpec(block, lambda i: (i, 0, 0)),
                pl.BlockSpec(block, lambda i: (i, 0, 0)),
            ],
            out_specs=pl.BlockSpec(block, lambda i: (i, 0, 0)),
        ),
    )(q, k, v)

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
    flops = 4 * B * H * S * S * D
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
