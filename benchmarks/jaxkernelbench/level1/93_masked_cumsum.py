"""
JAXBench Level 1 - Task 93: masked_cumsum
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.149306
"""

import jax
import jax.numpy as jnp

class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape).
            mask (jnp.ndarray): Boolean mask of the same shape as x.

        Returns:
            jnp.ndarray: Cumulative sum of elements where mask is True.
        """
        return jnp.cumsum(x * mask, axis=self.dim)

    def set_weights(self, weights_dict):
        # No learnable parameters, but including for consistency
        pass

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.uniform(key1, shape=(batch_size, *input_shape))
    mask = jax.random.bernoulli(key2, shape=(batch_size, *input_shape))
    return [x, mask]

def get_init_inputs():
    return [dim]