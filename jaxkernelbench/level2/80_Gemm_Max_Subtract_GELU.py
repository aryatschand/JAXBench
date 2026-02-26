"""
JAXBench Level 2 - Gemm_Max_Subtract_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu

class Model:
    def __init__(self, in_features, out_features, max_dim):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.max_dim = max_dim

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight) + self.bias
        x = jnp.max(x, axis=self.max_dim, keepdims=True)
        x = x - jnp.mean(x, axis=1, keepdims=True)
        x = gelu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, max_dim]