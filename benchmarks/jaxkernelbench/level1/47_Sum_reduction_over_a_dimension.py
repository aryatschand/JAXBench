"""
JAXBench Level 1 - Task 47: Sum_reduction_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.136904
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        self.dim = dim

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (jnp.ndarray): Input tensor of shape (..., dim, ...).

        Returns:
            jnp.ndarray: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return jnp.sum(x, axis=self.dim, keepdims=True)

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass

batch_size = 128
dim1 = 4096 
dim2 = 4095
reduce_dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [reduce_dim]