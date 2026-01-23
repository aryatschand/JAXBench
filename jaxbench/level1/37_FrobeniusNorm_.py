"""
JAXBench Level 1 - Task 37: FrobeniusNorm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.134193
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        pass

    def forward(self, x):
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x: Input JAX array of arbitrary shape.

        Returns:
            JAX array: Output array with Frobenius norm normalization applied, same shape as input.
        """
        norm = jnp.linalg.norm(x)
        return x / norm

    def set_weights(self, weights_dict):
        """Required empty method since model has no weights"""
        pass

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return []