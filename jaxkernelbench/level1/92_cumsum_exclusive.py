"""
JAXBench Level 1 - Task 92: cumsum_exclusive
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.148989
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        cumsum = jnp.cumsum(jax.lax.slice(x, 
            [0] * x.ndim, 
            [x.shape[i] if i != self.dim else x.shape[i]-1 for i in range(x.ndim)],
            [1] * x.ndim), 
            axis=self.dim)
        zeros = jnp.zeros_like(jax.lax.index_in_dim(x, 0, axis=self.dim, keepdims=True))
        return jnp.concatenate([zeros, cumsum], axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]