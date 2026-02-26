"""
JAXBench Level 1 - Task 34: InstanceNorm
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.133211
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        self.num_features = num_features
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            Output tensor with Instance Normalization applied, same shape as input.
        """
        # Calculate mean and variance along spatial dimensions
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        return x_norm

batch_size = 16  # Reduced from 112 for memory
features = 64
dim1 = 256  # Reduced from 512
dim2 = 256  # Reduced from 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]