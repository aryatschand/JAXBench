"""
JAXBench Level 1 - Task 40: LayerNorm
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.135170
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        self.normalized_shape = normalized_shape
        self.weight = jnp.ones(normalized_shape)
        self.bias = jnp.zeros(normalized_shape)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x: Input tensor of shape (*, normalized_shape).

        Returns:
            Output tensor with Layer Normalization applied, same shape as input.
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]