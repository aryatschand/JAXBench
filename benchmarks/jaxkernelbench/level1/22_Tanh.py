"""
JAXBench Level 1 - Task 22: Tanh
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.128938
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Tanh activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with Tanh applied, same shape as input.
        """
        return jnp.tanh(x)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed