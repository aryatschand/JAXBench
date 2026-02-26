"""
JAXBench Level 1 - Task 24: LogSoftmax
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.129719
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        self.dim = dim
    
    def forward(self, x):
        """
        Applies LogSoftmax activation to the input array.

        Args:
            x: Input array of shape (batch_size, dim).

        Returns:
            Output array with LogSoftmax applied, same shape as input.
        """
        return jax.nn.log_softmax(x, axis=self.dim)
    
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