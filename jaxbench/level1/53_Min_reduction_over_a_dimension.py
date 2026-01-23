"""
JAXBench Level 1 - Task 53: Min_reduction_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.139013
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        self.dim = dim

    def forward(self, x):
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x: Input JAX array.

        Returns:
            JAX array: Output array after min reduction over the specified dimension.
        """
        return jnp.min(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension