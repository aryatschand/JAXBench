"""
JAXBench Level 1 - Task 48: Mean_reduction_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.137298
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs mean reduction over a specific dimension.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x: Input array of arbitrary shape.

        Returns:
            Output array with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return jnp.mean(x, axis=self.dim)

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

batch_size = 128
dim1 = 4096 
dim2 = 4095

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1]