"""
JAXBench Level 1 - Task 90: cumprod
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.148274
"""

import jax
import jax.numpy as jnp

class Model:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape).

        Returns:
            jnp.ndarray: Array of the same shape as `x` after applying cumulative product along `dim`.
        """
        return jnp.cumprod(x, axis=self.dim)

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]