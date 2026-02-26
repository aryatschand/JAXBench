"""RMSNorm — Llama 3.1 8B. From openxla/tokamax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_8b_rmsnorm',
    'model': 'Llama-3.1-8B',
    'operator': 'rmsnorm',
    'shape': [1, 2048, 4096],
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, scale) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    shape = CONFIG['shape']
    x = jax.random.normal(k1, shape, dtype=dtype)
    scale = jax.random.normal(k2, (shape[-1],), dtype=dtype) * 0.1 + 1.0
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
    shape = CONFIG['shape']
    total_elements = 1
    for s in shape:
        total_elements *= s
    # Memory-bound: read x + scale, write output (bfloat16 = 2 bytes)
    total_bytes = total_elements * 2 * 2 + shape[-1] * 2
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
