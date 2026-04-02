```python
"""Grouped Matrix Multiply (Megablox GMM) — Qwen3-235B-A22B MoE dimensions.

Reference grouped matmul: for each expert group, slice the input tokens
and multiply with that expert's weight matrix. Core primitive for MoE layers.
From JAX experimental pallas ops (reference_gmm).

Not jit-compatible: uses data-dependent slicing on group_sizes.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'megablox_gmm_qwen3_235b',
    'model': 'Qwen3-235B-A22B',
    'operator': 'grouped_matmul',
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


def bmm_kernel(lhs_ref, rhs_ref, out_ref):
    K = lhs_ref.shape[1]
    BK = 512
    
    acc = jnp.zeros((lhs_ref.shape[0], out_ref.shape[1]), dtype=jnp.float32)
    
    def body(i, acc):
        l = lhs_ref[:, i * BK
