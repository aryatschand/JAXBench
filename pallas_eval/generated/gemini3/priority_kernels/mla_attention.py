```python
"""Multi-head Latent Attention (MLA) — DeepSeek V3 671B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

CONFIG = {
    'name': 'deepseek_v3_mla',
    'model': 'DeepSeek-V3-671B',
    'operator': 'mla_attention',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 7168,
    'num_heads': 128,
    'q_lora_rank': 1536,
    'kv_lora_rank': 512,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'rope_theta': 10000,
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, q_down, q_up, kv_down, k_up, v_up, o_proj)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = CONFIG
    B, S, E = C['batch'], C['seq_len'], C['emb_dim']
    H = C['num_heads']
    ql, kvl = C['q_lora_rank'], C['kv_lora_rank']
    nope, rope, vd = C['qk_nope_head_dim'], C['qk_rope_head_dim'],
