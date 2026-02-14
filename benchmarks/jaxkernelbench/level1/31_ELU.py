"""
JAXBench Level 1 - Task 31: ELU
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.131998
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        self.alpha = alpha
    
    def forward(self, x):
        """
        Applies ELU activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with ELU applied, same shape as input.
        """
        return jnp.where(x > 0, x, self.alpha * (jnp.exp(x) - 1))

    def set_weights(self, weights_dict):
        # No learnable parameters, but including for consistency
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization