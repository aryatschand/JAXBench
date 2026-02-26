"""
JAXBench Level 2 - Gemm_Sigmoid_Scaling_ResidualAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, input_size, hidden_size, scaling_factor):
        self.weight = jnp.zeros((input_size, hidden_size))
        self.bias = jnp.zeros(hidden_size)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight) + self.bias
        original_x = x
        x = jax.nn.sigmoid(x)
        x = x * self.scaling_factor
        x = x + original_x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]