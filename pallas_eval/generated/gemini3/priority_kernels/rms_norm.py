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

def workload(x, scale):
    B, S, D = x.shape
    x_2d = x.reshape((B * S, D))
    scale_2d = scale.reshape((1, D))
    
    eps = CONFIG['epsilon']
    
    def rmsnorm_kernel(x_ref, scale_ref, o_ref):
        x_val = x_ref[...]
        scale_val = scale_ref[...]
        
        x_f32 = x_val.astype(jnp.float32)
        x2 = lax.square(x_f32)
        mean2 = jnp.mean(x2, axis=-1, keepdims=True)
        rsqrt = lax.rsqrt(mean2 + eps)
        
        normed = x_f32 * rsqrt
        normed = normed.astype(x_val.dtype)
        
        scale_rep = pltpu.repeat(scale_val, x_val.shape[0], 0)
        o_ref[...] = normed * scale_rep

    block_S = 64
    grid = (B * S // block_S,)
    
    out_2d = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_S, D), lambda i: (i, 0)),
                pl.BlockSpec((1, D), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((block_S, D), lambda i: (i, 0)),
        ),
    )(x_2d, scale_2d)
    
    return out_2d.reshape((B, S, D))

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
