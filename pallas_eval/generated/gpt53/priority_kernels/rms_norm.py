"""RMSNorm — Llama-3.1-70B pre-attention normalization.

Root Mean Square Layer Normalization at Llama-3.1-70B scale.
Input shape: (batch=1, seq_len=2048, emb_dim=8192).
From MaxText layers/normalizations.py.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_rmsnorm',
    'model': 'Llama-3.1-70B',
    'operator': 'rms_norm',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 8192,
    'epsilon': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, scale) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    x = jax.random.normal(k1, (B, S, D), dtype=dtype)
    scale = jax.random.normal(k2, (D,), dtype=dtype) * 0.1 + 1.0
    return x, scale


def rmsnorm_kernel(x_ref, scale_ref, o_ref):
    x = x_ref[...]                     # (1, D)
    scale = scale_ref[...]             # (1, D)

    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(x_f32 * x_f32, axis=1, keepdims=True)
    norm = x_f32 * lax.rsqrt(mean2 + CONFIG['epsilon'])
    norm = jnp.asarray(norm, x.dtype)

    o_ref[...] = norm * scale


def workload(x, scale):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * scale"""
    B, S, D = x.shape
    x2d = jnp.reshape(x, (B * S, D))
    scale2d = jnp.reshape(scale, (1, D))

    block = (1, D)
    grid = (B * S,)

    out = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct((B * S, D), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i: (i, 0)),
                pl.BlockSpec(block, lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
        ),
    )(x2d, scale2d)

    return jnp.reshape(out, (B, S, D))


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
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    flops = 5 * B * S * D
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
