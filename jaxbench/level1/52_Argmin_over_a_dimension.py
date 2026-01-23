"""
JAXBench Level 1 - Task 52: Argmin_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.138697
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        self.dim = dim

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Array containing the indices of the minimum values along the specified dimension.
        """
        return jnp.argmin(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [dim]