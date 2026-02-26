"""
JAXBench Level 1 - Task 91: cumsum_reverse
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.148605
"""

import jax
import jax.numpy as jnp

class Model:
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        return jnp.flip(jnp.cumsum(jnp.flip(x, axis=self.dim), axis=self.dim), axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]