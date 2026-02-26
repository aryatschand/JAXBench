"""RMSNorm — Llama 3.1 70B. Based on MaxText normalizations.py."""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'llama3_70b_rmsnorm',
    'model': 'Llama-3.1-70B',
    'operator': 'rmsnorm',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 8192,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, scale) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    B, S, E = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    x = jax.random.normal(k1, (B, S, E), dtype=dtype)
    scale = jax.random.normal(k2, (E,), dtype=dtype) * 0.1 + 1.0
    return x, scale


def workload(x, scale):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * scale"""
    eps = CONFIG['eps']
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * scale


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
    B, S, E = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    total_bytes = B * S * E * 2 * 2 + E * 2
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(total_bytes / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
