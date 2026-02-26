"""
JAXBench Level 1 - Task 28: HardSigmoid
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.130975
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies HardSigmoid activation to the input tensor.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with HardSigmoid applied, same shape as input.
        """
        return jnp.clip(jnp.clip(x + 3.0, a_min=0.0) / 6.0, a_max=1.0)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed