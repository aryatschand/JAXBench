```python
"""Vanilla Multi-Head Causal Attention — Baseline (64 heads, seq=2048).

Standard scaled dot-product attention with causal mask.
No GQA, no softcap, no sliding window — pure MHA baseline.
Matches Pallas flash_attention kernel config.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'flash_attention_baseline',
    'model': 'Baseline-MHA',
    'operator': 'causal_mha',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    value = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return query, key_t, value


def attention_kernel(q_ref, k_ref, v_ref, o_ref):
    q_idx = pl.program_id(2)
    
    # Load blocks into VMEM
    q = q_ref[0, 0, :, :]
    k = k_ref[0, 0, :, :]
    v = v_ref[0, 0, :, :]
    
    Q_BLOCK = q.shape[0]
    S = k.shape[0]
    
    scale = 1.0 / jnp.sqrt(q.shape[1])
    
    # QK^T
    qk = jax.lax.dot_general(
        q, k,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32
    ) * scale
    
    # Causal mask
    row_idx = q_idx * Q_BLOCK + jnp.arange(Q_BLOCK)
    col_idx = jnp.arange(S)
    mask = row_idx[:, None] >= col_idx[None, :]
    
    qk = jnp.where
