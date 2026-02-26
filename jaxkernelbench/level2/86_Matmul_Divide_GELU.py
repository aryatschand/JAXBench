"""
JAXBench Level 2 - Matmul_Divide_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn

class Model:
    def __init__(self, input_size, output_size, divisor):
        self.weight = jnp.zeros((input_size, output_size))
        self.bias = jnp.zeros(output_size)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = x @ self.weight + self.bias
        x = x / self.divisor
        x = jnn.gelu(x)
        return x

batch_size = 1024
input_size = 8192 
output_size = 8192
divisor = 10.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, output_size, divisor]