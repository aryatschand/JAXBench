"""
JAXBench Level 1 - Task 21: Sigmoid
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.128539
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a Sigmoid activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Sigmoid activation to the input tensor.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with Sigmoid applied, same shape as input.
        """
        return jax.nn.sigmoid(x)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed