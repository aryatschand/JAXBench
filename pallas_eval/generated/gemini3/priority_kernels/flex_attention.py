```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_flex_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'flex_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, k, v, rel_pos_bias) tensors."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype) * 0.02
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype) * 0.02
    # Relative position bias per head (learned, as in ALiBi-style)
    rel_pos_bias = jax.random.normal(k4, (H, S, S), dtype=dtype) * 0.01
    return q, k, v, rel_pos_bias


def flex_attention_kernel(q_ref, k_ref, v_ref, rel_pos_bias_ref, o_ref):
    q_idx = pl.program_id(2)
    B_Q = q_ref.shape[0]
    D = q_ref.shape[1]
    S = k_ref.shape[0]
    sm_scale = D ** -0.5
    
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    rpb = rel_pos_bias_ref[...]
    
    # Compute QK^T
    qk = jax.lax.dot_general(
        q, k,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32
    ) * sm_scale
    
    # Add relative position bias
    qk = qk + rpb
    
    # Causal mask
    row_idx = q_idx * B_Q + jnp
