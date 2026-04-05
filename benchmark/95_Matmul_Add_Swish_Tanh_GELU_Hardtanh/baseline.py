"""
JAXBench Level 2 - Matmul_Add_Swish_Tanh_GELU_Hardtanh
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, add_value_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.add_value = jnp.zeros(add_value_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = x @ self.weight + self.bias
        x = x + self.add_value
        x = jax.nn.swish(x)  # Swish
        x = jnp.tanh(x)
        x = jax.nn.gelu(x)  # GELU
        x = jnp.clip(x, -1.0, 1.0)  # Hardtanh
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]