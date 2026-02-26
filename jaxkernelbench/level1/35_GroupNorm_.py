"""
JAXBench Level 1 - Task 35: GroupNorm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.133536
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        self.num_groups = num_groups
        self.num_features = num_features
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Applies Group Normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_features, *).

        Returns:
            Output tensor with Group Normalization applied, same shape as input.
        """
        input_shape = x.shape
        batch_size = input_shape[0]
        num_channels = input_shape[1]
        
        # Reshape input: (N, C, *) -> (N * G, C // G, *)
        x = x.reshape((batch_size, self.num_groups, -1))
        
        # Calculate mean and var
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + self.eps)
        
        # Reshape back
        x = x.reshape(input_shape)
        
        # Apply scale and shift
        return x * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)

batch_size = 16  # Reduced from 112 for memory
features = 64
num_groups = 8
dim1 = 256  # Reduced from 512
dim2 = 256  # Reduced from 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features, num_groups]