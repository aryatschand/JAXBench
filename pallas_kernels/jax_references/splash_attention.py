"""JAX reference for Pallas splash attention — Llama-3.1-70B GQA dimensions.

Adapted from _attention_reference_default + make_attention_reference in
jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py.

Uses jax.vmap over KV heads and Q heads per KV group, matching upstream.
"""

import numpy as np

import jax
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

CONFIG = {
    'name': 'pallas_splash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_splash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H_q, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H_kv, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H_kv, S, D), dtype=dtype)
    return q, k, v


def _attention_reference_default(mask, q, k, v):
    """Single-head attention reference from upstream splash_attention_kernel.py."""
    logits = jnp.einsum(
        "sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32)
    )
    logits = jnp.where(mask, logits, DEFAULT_MASK_VALUE)
    m = logits.max(axis=-1)
    s = jnp.exp(logits - m[..., None])
    l = s.sum(axis=-1)
    s = s / l[..., None]
    return jnp.einsum("st,td->sd", s, v.astype(jnp.float32))


def workload(q, k, v):
    """GQA reference using upstream vmap-over-heads pattern."""
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]

    causal_mask = np.tril(np.ones((S, S), dtype=np.bool_))

    def _single_batch(q_b, k_b, v_b):
        q_heads_per_kv = H_q // H_kv
        q_reshaped = q_b.reshape(H_kv, q_heads_per_kv, S, D)
        mask_reshaped = np.broadcast_to(causal_mask, (H_kv, q_heads_per_kv, S, S))

        fn = _attention_reference_default
        fn = jax.vmap(fn, in_axes=(0, 0, None, None))
        fn = jax.vmap(fn, in_axes=(0, 0, 0, 0))

        out = fn(mask_reshaped, q_reshaped, k_b, v_b)
        return out.reshape(H_q, S, D)

    results = []
    for b in range(B):
        results.append(_single_batch(q[b], k[b], v[b]))
    return jnp.stack(results, axis=0)
