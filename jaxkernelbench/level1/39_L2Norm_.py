"""
JAXBench Level 1 - Task 39: L2Norm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.134901
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        pass

    def forward(self, x):
        """
        Applies L2 normalization to the input tensor.

        Args:
            x: Input array of shape (*, dim, *).

        Returns:
            Output array with L2 normalization applied, same shape as input.
        """
        return x / jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)

    def set_weights(self, weights_dict):
        # No weights to set
        pass

batch_size = 4096  # Reduced from 32768 for memory
dim = 8192  # Reduced from 65535

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []