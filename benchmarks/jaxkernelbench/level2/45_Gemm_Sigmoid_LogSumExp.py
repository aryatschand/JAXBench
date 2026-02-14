"""
JAXBench Level 2 - Gemm_Sigmoid_LogSumExp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with same shapes as PyTorch
        self.linear1_weight = jnp.zeros((hidden_size, input_size))
        self.linear1_bias = jnp.zeros(hidden_size)
        self.linear2_weight = jnp.zeros((output_size, hidden_size))
        self.linear2_bias = jnp.zeros(output_size)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # First linear layer
        x = jnp.matmul(x, self.linear1_weight.T) + self.linear1_bias

        # Sigmoid activation
        x = jax.nn.sigmoid(x)

        # Second linear layer
        x = jnp.matmul(x, self.linear2_weight.T) + self.linear2_bias

        # LogSumExp over features
        x = jax.nn.logsumexp(x, axis=1)
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, output_size]