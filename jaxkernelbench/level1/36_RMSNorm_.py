"""
JAXBench Level 1 - Task 36: RMSNorm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.133895
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x: Input array of shape (batch_size, num_features, *).

        Returns:
            Output array with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=1, keepdims=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms

    def set_weights(self, weights_dict):
        # No learnable parameters for this model
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
    return [features]