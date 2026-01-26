"""
JAXBench Level 2 - Gemm_Scale_BatchNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 39: Gemm_Scale_BatchNorm
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Linear layer parameters
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        self.gemm_weight = jax.random.normal(key1, (out_features, in_features))
        self.gemm_bias = jax.random.normal(key2, (out_features,))

        # Scale parameter
        self.scale = jax.random.normal(key3, scale_shape)

        # BatchNorm1d parameters (learnable only)
        self.bn_weight = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features).
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Linear: x @ weight.T + bias
        x = jnp.matmul(x, self.gemm_weight.T) + self.gemm_bias

        # Scale
        x = x * self.scale

        # BatchNorm1d (training mode - use batch statistics)
        mean = jnp.mean(x, axis=0, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps) * self.bn_weight + self.bn_bias

        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_features))
    return [x]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
