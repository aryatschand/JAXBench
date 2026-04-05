"""
JAXBench Level 2 - Matmul_Sigmoid_Sum
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, input_size, hidden_size):
        self.weight = jnp.zeros((input_size, hidden_size))
        self.bias = jnp.zeros(hidden_size)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, input_size).

        Returns:
            Output array of shape (batch_size, 1).
        """
        x = jnp.matmul(x, self.weight) + self.bias
        x = jax.nn.sigmoid(x)
        x = jnp.sum(x, axis=1, keepdims=True)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size]