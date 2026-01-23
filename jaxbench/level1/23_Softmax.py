"""
JAXBench Level 1 - Task 23: Softmax
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.129344
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Softmax activation to the input tensor.

        Args:
            x: Input array of shape (batch_size, num_features).

        Returns:
            Output array with Softmax applied, same shape as input.
        """
        return jax.nn.softmax(x, axis=1)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed