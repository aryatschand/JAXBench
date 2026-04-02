```python
"""Mamba-2 State Space Duality (SSD) — Dao & Gu.

The SSD layer shows that structured state space models are equivalent to a form
of linear attention with input-dependent (selective) decay. This is the matrix
(parallel) form of Mamba-2's core computation.

Paper: "Transformers are SSMs" (Dao & Gu, 2024)
Mamba-2 is the dominant alternative to standard transformers in 2024-2025.

Config based on Mamba-2-2.7B from the paper.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

CONFIG = {
    'name': 'mamba2_2_7b_ssd',
    'model': 'Mamba-2-2.7B',
    'operator': 'state_space_duality',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 64,
    'd_state': 128,
    'd_model': 2560,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, A_log)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    # In Mamba-2 SSD: C maps to Q, B maps to K, x maps to V
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)  # C (output projection)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)  # B (input projection)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)  # x (hidden state)
    # A: input-dependent decay (after log-space parameterization)
    # Initialized negative (stable decay), per-head scalar
    A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
    return query, key_t, value, A_log


def ssd_kernel(q_ref, k_ref, v_ref, la_q_ref, la_k_ref, o_ref):
    Q_BLOCK = q
