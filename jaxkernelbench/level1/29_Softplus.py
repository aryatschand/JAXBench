"""
JAXBench Level 1 - Task 29: Softplus
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.131254
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a Softplus activation.
    """
    def __init__(self):
        pass

    def forward(self, x):
        """
        Applies Softplus activation to the input tensor.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with Softplus applied, same shape as input.
        """
        return jnp.logaddexp(x, 0.0)

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed