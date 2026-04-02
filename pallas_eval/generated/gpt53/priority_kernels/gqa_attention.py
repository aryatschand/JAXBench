"""Grouped Query Attention (GQA) — Llama 3.1 405B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_405b_gqa',
    'model': 'Llama-3.1-405B',
    'operator': 'gqa_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 128,
    'num_kv_heads': 8,
    'head_dim': 128,
    'emb_dim': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    Hq, Hkv, D = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=dtype)
    return query, key_t, value


def gqa_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[...]  # (S, D)
    k = k_ref[...]  # (S, D)
    v = v_ref[...]  # (S, D)

    D = q.shape[1]
    scale = jnp.float32(D) ** -0.5

    attn = jnp.matmul(q.astype(jnp.float32), k.T.astype(jnp.float32)) * scale

    S = q.shape[0]
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(mask, attn, -1e9)

    attn = jax.nn.softmax(attn, axis=-1).astype(q.dtype)
    out = jnp.matmul(attn.astype(jnp.float32), v.astype(jnp.float32)).astype(q.dtype)

    o_ref[...] = out


def workload(query, key, value):
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv

    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)

    q = query.transpose(0, 2, 1, 3).reshape(B * Hq, S, D)
    k = key.transpose(0, 2, 1, 3).reshape(B * Hq, S, D)
    v = value.transpose(0, 2, 1, 3).reshape(B * Hq, S, D)

    BH = B * Hq

    out = pl.pallas_call(
        gqa_kernel,
        out_shape=jax.ShapeDtypeStruct((BH, S, D), q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(BH,),
            in_specs=[
                pl.BlockSpec((S, D), lambda i: (i, 0, 0)),
                pl.BlockSpec((S, D), lambda i: (i, 0, 0)),
                pl.BlockSpec((S, D), lambda i: (i, 0, 0)),
            ],
            out_specs=pl.BlockSpec((S, D), lambda i: (i, 0, 0)),
        ),
    )(q, k, v)

    out = out.reshape(B, Hq, S, D).transpose(0, 2, 1, 3)
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
    B, S, Hq, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_query_heads'], CONFIG['head_dim']
    flops = B * Hq * S * S * D * 4
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
