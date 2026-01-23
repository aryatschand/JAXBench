"""
JAXBench Level 1 - Task 19: ReLU
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.127693
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies ReLU activation to the input array.

        Args:
            x: Input JAX array of any shape.

        Returns:
            JAX array: Output array with ReLU applied, same shape as input.
        """
        return jnp.maximum(0, x)

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