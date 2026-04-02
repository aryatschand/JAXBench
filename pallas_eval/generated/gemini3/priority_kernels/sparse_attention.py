```python
"""Sparse (Splash) Attention — Llama-3.1-70B GQA with causal mask.

Pallas TPU kernel implementation for sparse/splash attention: standard dot-product
attention with causal masking and grouped-query attention (GQA).
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_sparse_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'sparse_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, k, v) tensors in [num_heads, seq_len, head_dim] layout."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    S = CONFIG['seq_len']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (H_q, S, D), dtype=dtype) * (D ** -0.5)
    k = jax.random.normal(k2, (H_kv, S, D), dtype=dtype) * 0.02
    v = jax.random.normal(k3, (H_kv, S, D), dtype=dtype) * 0.02
    return q, k, v


def attn_kernel(q_ref, k_ref, v_ref, o_ref):
    i = pl.program_id(1)
    
    B_Q = 256
    S = 2048
    
    # Read entire blocks and squeeze the head dimension
    q = q_ref[...][0]  # (B_Q, D)
    k = k_ref[...][0]  # (S, D)
    v = v_ref[...][0]  # (S, D)
    
    # Attention scores: (B_Q, D) x (S, D) -> (B_Q, S)
    attn = jax.lax.dot_general(
        q, k,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32
    )
