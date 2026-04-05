"""
JAXBench Level 2 - Gemm_Add_ReLU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, bias_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input array with shape (batch_size, in_features)
        Returns:
            Output array with shape (batch_size, out_features)
        """
        x = jnp.matmul(x, self.weight)
        x = x + self.bias
        x = jax.nn.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bias_shape]