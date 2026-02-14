"""
JAXBench Level 2 - Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros(out_features)
        else:
            self.bias = None
        self.subtract = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        original_x = x

        # Gemm
        x = jnp.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias

        # Subtract
        x = x - self.subtract

        # GlobalAvgPool
        x = jnp.mean(x, axis=1, keepdims=True)

        # LogSumExp
        x = jax.nn.logsumexp(x, axis=1, keepdims=True)

        # GELU
        x = jax.nn.gelu(x)

        # ResidualAdd
        x = x + original_x

        return x

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]