"""
JAXBench Level 2 - Matmul_Min_Subtract
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, constant):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.constant = jnp.array(constant)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = x @ self.weight + self.bias
        x = jnp.minimum(x, self.constant)
        x = x - self.constant
        return x

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, constant]