"""Linear Attention via FAVOR+ (Performer) — google-research/performer.

Replaces O(n^2) softmax attention with O(n) linear attention using random
feature maps. Uses nonnegative softmax kernel features (FAVOR+).

Source: https://github.com/google-research/google-research/tree/master/performer
Paper: "Rethinking Attention with Performers" (Choromanski et al., 2021)
"""
import math
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'performer_favor_attention',
    'model': 'Performer',
    'operator': 'favor_plus_linear_attention',
    'batch': 1,
    'seq_len': 4096,
    'num_heads': 16,
    'head_dim': 64,
    'nb_features': 256,
}


def _create_projection_matrix(key, nb_features, head_dim):
    """Gaussian orthogonal random matrix via stacked QR decomposition."""
    nb_full_blocks = nb_features // head_dim
    remainder = nb_features - nb_full_blocks * head_dim
    blocks = []
    for i in range(nb_full_blocks):
        k = jax.random.fold_in(key, i)
        G = jax.random.normal(k, (head_dim, head_dim))
        Q, _ = jnp.linalg.qr(G)
        blocks.append(Q)
    if remainder > 0:
        k = jax.random.fold_in(key, nb_full_blocks)
        G = jax.random.normal(k, (head_dim, head_dim))
        Q, _ = jnp.linalg.qr(G)
        blocks.append(Q[:remainder])
    projection = jnp.concatenate(blocks, axis=0)
    # Rescale rows by norms of iid Gaussian rows
    k2 = jax.random.fold_in(key, nb_full_blocks + 1)
    norms = jnp.linalg.norm(
        jax.random.normal(k2, (nb_features, head_dim)), axis=1
    )
    return projection * norms[:, None]


def _favor_features(data, projection_matrix, is_query):
    """FAVOR+ nonnegative softmax kernel features.

    data: (B, H, S, D)
    projection_matrix: (M, D)
    Returns: (B, H, S, M)
    """
    d = data.shape[-1]
    m = projection_matrix.shape[0]
    data_normalizer = 1.0 / (d ** 0.25)
    ratio = 1.0 / math.sqrt(m)
    data_dash = jnp.einsum('...d,md->...m', data * data_normalizer, projection_matrix)
    diag_data = jnp.sum((data * data_normalizer) ** 2, axis=-1, keepdims=True) / 2.0
    if is_query:
        stab = jnp.max(data_dash, axis=-1, keepdims=True)
    else:
        stab = jnp.max(data_dash)
    return ratio * jnp.exp(data_dash - diag_data - stab + 1e-6)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, projection_matrix)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    M = CONFIG['nb_features']
    query = jax.random.normal(k1, (B, S, H, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, H, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, H, D), dtype=dtype)
    proj = _create_projection_matrix(k4, M, D).astype(dtype)
    return query, key_t, value, proj


def workload(query, key, value, projection_matrix):
    """Bidirectional linear attention via FAVOR+.

    O(n * m * d) instead of O(n^2 * d) where m = nb_features << n.
    """
    B, S, H, D = query.shape
    q = query.transpose(0, 2, 1, 3)  # (B, H, S, D)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    # Map Q, K to random feature space
    q_prime = _favor_features(q, projection_matrix, is_query=True)   # (B, H, S, M)
    k_prime = _favor_features(k, projection_matrix, is_query=False)  # (B, H, S, M)
    # Linear attention: Q'(K'^T V) instead of softmax(QK^T)V
    kv = jnp.einsum('bhsm,bhsd->bhmd', k_prime, v)  # (B, H, M, D)
    qkv = jnp.einsum('bhsm,bhmd->bhsd', q_prime, kv)  # (B, H, S, D)
    # Normalize
    k_sum = jnp.sum(k_prime, axis=2)  # (B, H, M)
    normalizer = jnp.einsum('bhsm,bhm->bhs', q_prime, k_sum)  # (B, H, S)
    out = qkv / (normalizer[..., None] + 1e-6)
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
    M = CONFIG['nb_features']
    # Feature map: 2 * B*H*S*(D*M) for Q' and K'
    # KV product: 2 * B*H*S*M*D
    # QKV product: 2 * B*H*S*M*D
    # Normalizer: 2 * B*H*S*M
    flops = 2 * B * H * S * D * M * 2 + 2 * B * H * S * M * D * 2
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
