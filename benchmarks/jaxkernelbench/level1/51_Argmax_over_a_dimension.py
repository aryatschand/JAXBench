"""
JAXBench Level 1 - Task 51: Argmax_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.138352
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        self.dim = dim

    def forward(self, x):
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x: Input JAX array.

        Returns:
            JAX array: Output array with argmax applied, with the specified dimension removed.
        """
        return jnp.argmax(x, axis=self.dim)

    def set_weights(self, weights_dict):
        # No weights to set for this model
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