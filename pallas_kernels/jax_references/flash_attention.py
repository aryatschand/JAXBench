"""JAX reference for Pallas flash attention — Llama-3.1-70B scale MHA.

Adapted from mha_reference_no_custom_vjp in
jax/experimental/pallas/ops/tpu/flash_attention.py.
"""

import jax
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

CONFIG = {
    'name': 'pallas_flash_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_flash_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    """Upstream mha_reference_no_custom_vjp with causal=True."""
    sm_scale = 1.0 / (CONFIG['head_dim'] ** 0.5)
    mask_value = DEFAULT_MASK_VALUE

    logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    logits *= sm_scale

    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]

    logits = logits + jnp.where(causal_mask, 0.0, mask_value)

    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    return jnp.einsum("bhqk,bhkc->bhqc", weights, v)
