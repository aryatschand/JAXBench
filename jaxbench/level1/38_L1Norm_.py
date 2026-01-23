"""
JAXBench Level 1 - Task 38: L1Norm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.134477
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        pass

    def forward(self, x):
        """
        Applies L1 normalization to the input tensor.

        Args:
            x: Input array of shape (..., dim, ...).

        Returns:
            Output array with L1 normalization applied, same shape as input.
        """
        return x / jnp.mean(jnp.abs(x), axis=1, keepdims=True)

    def set_weights(self, weights_dict):
        """
        No weights to set for this model.
        """
        pass

batch_size = 4096  # Reduced from 32768 for memory
dim = 8192  # Reduced from 65535

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []