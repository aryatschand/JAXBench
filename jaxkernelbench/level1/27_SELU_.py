"""
JAXBench Level 1 - Task 27: SELU_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.130719
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a SELU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies SELU activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with SELU applied, same shape as input.
        """
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

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