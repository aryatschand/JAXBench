"""Flash Attention (Optimized) — Baseline MHA with jax.nn.dot_product_attention.

Uses JAX's built-in dot_product_attention which dispatches to flash attention
on TPU/GPU backends, avoiding O(S^2) memory materialization.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'flash_attention_optimized',
    'model': 'Baseline-MHA',
    'operator': 'causal_mha',
    'batch': 1,
    'seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value) in (B, H, S, D) layout."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(query, key, value):
    """Optimized causal MHA using jax.nn.dot_product_attention.

    dot_product_attention expects (B, S, H, D) layout, so we transpose
    from the standard (B, H, S, D) used by the baseline.
    """
    # (B, H, S, D) -> (B, S, H, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
    # (B, S, H, D) -> (B, H, S, D)
    return out.transpose(0, 2, 1, 3)


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
