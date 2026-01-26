"""
JAXBench Level 2 - Gemm_Swish_Divide_Clamp_Tanh_Clamp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        x = x * jax.nn.sigmoid(x)  # Swish activation
        x = x / 2.0
        x = jnp.clip(x, -1.0, 1.0)  # Clamp between -1 and 1
        x = jnp.tanh(x)  # Tanh activation
        x = jnp.clip(x, -1.0, 1.0)  # Clamp between -1 and 1
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]