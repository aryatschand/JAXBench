```python
"""Grouped Matmul (Ragged Dot) for MoE — Mixtral 8x7B. From openxla/tokamax."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'mixtral_8x7b_ragged_dot',
    'model': 'Mixtral-8x7B',
    'operator': 'ragged_dot',
    'num_groups': 8,
    'M': 8192,
    'K': 4096,
    'N': 14336,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, weights, group_sizes) for grouped matmul."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    G, M, K, N = CONFIG['num_groups'], CONFIG['M'], CONFIG['K'], CONFIG['N']
    # Each group gets M/G rows
    x = jax.random.normal(k1, (G, M // G, K), dtype=dtype)
    weights = jax.random.normal(k2, (G, K, N), dtype=dtype) * 0.02
    return x, weights


def ragged_dot_kernel(x_ref, w_ref, o_ref):
    # Read entire blocks into VMEM
    x_val = x_ref[...]  # Shape: (1, BM, K)
    w_val = w_ref[...]  # Shape: (1, K, BN)
    
    # Perform batched matmul (batch size 1). Mosaic will automatically tile this 
    # over the MXU since the K dimension is fully loaded in VMEM.
    o_val = jnp.matmul(x_val, w_val, preferred_element_type=jnp.float32)
    
    # Write the result back to HBM
    o_ref[...] = o_val.astype(o_ref.dtype)


def workload(x, weights):
    """Grouped matmul: each group does independent matmul. Equivalent to ragged dot."""
    G, M_g, K = x.shape
    _, _, N = weights.shape
    
    # Block sizes chosen to maximize MXU utilization while fitting comfortably in VMEM.
    # x_block: 1 * 256 * 4096 * 2B = 2 MB
    # w_block: 1 * 4096 * 512 * 2B = 4 MB
    # o_block: 1 * 2
