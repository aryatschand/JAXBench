"""
JAXBench Level 1 - Task 89: cumsum
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.147886
"""

import jax
import jax.numpy as jnp

class Model:
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape), where `*input_shape` 
                            can vary depending on the use case.

        Returns:
            jnp.ndarray: Array of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return jnp.cumsum(x, axis=self.dim)

    def set_weights(self, weights_dict):
        """
        No-op since this model has no learnable parameters.
        """
        pass

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated array with shape 
              (batch_size, *input_shape).
    """
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]