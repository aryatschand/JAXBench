"""
JAXBench Level 2 - Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu

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
        # Gemm
        x = jnp.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias

        # LogSumExp
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)

        # LeakyReLU
        x = jnp.where(x > 0, x, 0.01 * x)

        # LeakyReLU
        x = jnp.where(x > 0, x, 0.01 * x)

        # GELU
        x = gelu(x)

        # GELU
        x = gelu(x)

        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]