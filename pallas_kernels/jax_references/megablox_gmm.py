"""JAX reference for Pallas megablox GMM — Qwen3-235B-A22B MoE dimensions.

Adapted from reference_gmm in jax/tests/pallas/tpu_gmm_test.py.
Not jit-compatible: uses data-dependent slicing on group_sizes.
"""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'pallas_megablox_gmm_qwen3_235b',
    'model': 'Qwen3-235B-A22B',
    'operator': 'pallas_gmm',
    'num_experts': 128,
    'num_experts_per_tok': 8,
    'emb_dim': 4096,
    'moe_mlp_dim': 1536,
    'seq_len': 2048,
}

_skip_jit = True


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    G = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    K = CONFIG['emb_dim']
    N = CONFIG['moe_mlp_dim']
    S = CONFIG['seq_len']
    M = S * top_k
    limit = 1 / (M * K)
    lhs = jax.random.uniform(k1, (M, K), dtype=dtype, minval=-limit, maxval=limit)
    lhs = lhs.astype(jnp.bfloat16).astype(dtype)
    rhs = jax.random.uniform(k2, (G, K, N), dtype=dtype, minval=-limit, maxval=limit)
    rhs = rhs.astype(jnp.bfloat16).astype(dtype)
    tokens_per_expert = M // G
    group_sizes = jnp.full((G,), tokens_per_expert, dtype=jnp.int32)
    return lhs, rhs, group_sizes


def workload(lhs, rhs, group_sizes):
    """Reference grouped matmul from upstream JAX tests.

    For each group i, slices lhs[start:start+size] and computes dot with rhs[i].
    Uses data-dependent slicing so must be run eagerly (not under jax.jit).
    """
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
        result = jax.lax.dot(
            lhs[start:start + size, :],
            rhs[i, :, :],
            preferred_element_type=jnp.float32,
        )
        out.append(result)
        start += group_sizes[i]
    return jnp.concatenate(out, axis=0)
