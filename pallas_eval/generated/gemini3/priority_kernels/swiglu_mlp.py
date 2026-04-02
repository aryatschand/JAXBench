"""SwiGLU MLP — Llama 3.1 70B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_swiglu',
    'model': 'Llama-3.1-70B',
    'operator': 'swiglu_mlp',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 8192,
    'mlp_dim': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, gate_kernel, up_kernel, down_kernel)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B, S, E, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['mlp_dim']
    x = jax.random.normal(k1, (B, S, E), dtype=dtype)
    gate = jax.random.normal(k2, (E, M), dtype=dtype) * 0.02
    up = jax.random.normal(k3, (E, M), dtype=dtype) * 0.02
    down = jax.random.normal(k4, (M, E), dtype=dtype) * 0.02
    return x, gate, up, down


def swiglu_kernel(gate_ref, up_ref, o_ref):
    g = gate_ref[...]
    u = up_ref[...]
    o_ref[...] = jax.nn.silu(g) * u


def pallas_swiglu(gate, up):
    B, S, M = gate.shape
    gate_2d = gate.reshape(B * S, M)
    up_2d = up.reshape(B * S, M)
    
    block_m = 256
    block_n = 1024
    
    grid = (gate_2d.shape[0] // block_m, gate_2d.shape[1] // block_n)
    
    out_2d = pl.pallas_call(
        swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct(gate_2d.shape, gate_2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        )
    )(gate_2d, up_2d)
    
    return out_2d.reshape(B, S, M)


def workload(x, gate_kernel, up_kernel, down_kernel):
    """SwiGLU: output = (SiLU(x @ gate) * (x @ up)) @ down"""
    gate = jnp.dot(x, gate_kernel)
    up = jnp.dot(x, up_kernel)
    hidden = pallas_swiglu(gate, up)
    return jnp.dot(hidden, down_kernel)


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
    B, S, E, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['mlp_dim']
    flops = B * S * E * M * 2 * 3
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
