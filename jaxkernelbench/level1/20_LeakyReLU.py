"""
JAXBench Level 1 - Task 20: LeakyReLU
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.128042
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        self.negative_slope = negative_slope
    
    def forward(self, x):
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with LeakyReLU applied, same shape as input.
        """
        return jnp.where(x > 0, x, x * self.negative_slope)

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