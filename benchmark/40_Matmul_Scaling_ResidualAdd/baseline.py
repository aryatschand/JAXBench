"""
JAXBench Level 2 - Matmul_Scaling_ResidualAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, scaling_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight) + self.bias
        original_x = x
        x = x * self.scaling_factor
        x = x + original_x
        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]