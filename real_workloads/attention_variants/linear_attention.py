"""Causal Linear Attention — Katharopoulos et al.

Foundation of all recent linear attention work (RetNet, GLA, RWKV, Mamba-2).
Replaces softmax with a kernel feature map φ(x) = elu(x) + 1, enabling O(n)
recurrent computation via the associative property.

Paper: "Transformers are RNNs" (Katharopoulos et al., ICML 2020)
Still heavily relevant in 2024-2025 as the basis for GLA, RetNet, Based, etc.

Config based on a 1.3B-scale model (similar to recent linear attention papers).
Uses the parallel quadratic form as the baseline to optimize with Pallas.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'causal_linear_attention',
    'model': 'LinearAttn-1.3B',
    'operator': 'causal_linear_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 16,
    'head_dim': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """Causal linear attention with elu+1 kernel feature map.

    Parallel form: O_i = φ(Q_i) * Σ_{j≤i} [φ(K_j)^T V_j]
    Implemented as cumulative sum of KV outer products for causality.
    The normalizer ensures outputs are properly scaled.
    """
    # Feature map: φ(x) = elu(x) + 1
    phi_q = jax.nn.elu(query) + 1.0   # (B, H, S, D)
    phi_k = jax.nn.elu(key) + 1.0     # (B, H, S, D)

    # KV state: cumulative sum of outer products φ(K)^T ⊗ V
    # S_i = Σ_{j≤i} φ(K_j) ⊗ V_j  ->  (B, H, S, D, Dv)
    kv = jnp.einsum('bhsd,bhsv->bhsdv', phi_k, value)
    kv_cumsum = jnp.cumsum(kv, axis=2)

    # Numerator: φ(Q) @ S
    numerator = jnp.einsum('bhsd,bhsdv->bhsv', phi_q, kv_cumsum)

    # Denominator (normalizer): φ(Q) @ Σ_{j≤i} φ(K_j)
    k_cumsum = jnp.cumsum(phi_k, axis=2)  # (B, H, S, D)
    denominator = jnp.einsum('bhsd,bhsd->bhs', phi_q, k_cumsum)  # (B, H, S)
    denominator = jnp.maximum(denominator, 1e-6)

    output = numerator / denominator[..., None]
    return output


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
    # KV outer products: B*H*S*D*D (einsum), cumsum: B*H*S*D*D
    # QKV product: B*H*S*D*D, normalizer: B*H*S*D
    flops = 2 * B * H * S * D * D + 2 * B * H * S * D * D + 2 * B * H * S * D
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
