"""RoPE — Llama 4 Scout 109B. Based on AI-Hypercomputer/maxtext.

Llama 4 uses RoPE with theta=500,000 (same as Llama 3.1) but with
48 query heads and head_dim=128. The larger head count creates a
different computational profile for the position embedding operation.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'llama4_scout_rope',
    'model': 'Llama-4-Scout-109B',
    'operator': 'rope',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 48,
    'head_dim': 128,
    'rope_theta': 500000,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x,) — input tensor to apply RoPE to."""
    key = jax.random.PRNGKey(42)
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    x = jax.random.normal(key, (B, S, H, D), dtype=dtype)
    return (x,)


def workload(x):
    """Apply RoPE: rotate pairs of dimensions by position-dependent angles."""
    B, S, H, D = x.shape
    theta = CONFIG['rope_theta']
    dim_pairs = D // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))
    positions = jnp.arange(S, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)
    cos = jnp.cos(angles).astype(x.dtype)[None, :, None, :]
    sin = jnp.sin(angles).astype(x.dtype)[None, :, None, :]
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


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
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    # Elementwise: 6 ops per element (2 mul + 2 mul + add + sub per pair)
    flops = B * S * H * D * 6
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
