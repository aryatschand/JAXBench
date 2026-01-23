"""
JAXBench Level 1 - Task 26: GELU_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.130453
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies GELU activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with GELU applied, same shape as input.
        """
        return jax.nn.gelu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed