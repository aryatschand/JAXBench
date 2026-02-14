"""
JAXBench Level 1 - Task 32: HardTanh
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.132469
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies HardTanh activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with HardTanh applied, same shape as input.
        """
        return jnp.clip(x, -1.0, 1.0)

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