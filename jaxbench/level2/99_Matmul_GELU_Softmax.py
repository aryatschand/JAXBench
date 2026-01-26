"""
JAXBench Level 2 - Matmul_GELU_Softmax
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu, softmax

class Model:
    def __init__(self, in_features, out_features):
        # Initialize with same shapes as PyTorch
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight) + self.bias
        x = gelu(x)
        x = softmax(x, axis=1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]