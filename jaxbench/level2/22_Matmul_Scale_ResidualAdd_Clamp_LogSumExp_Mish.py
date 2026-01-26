"""
JAXBench Level 2 - Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        self.weight = jnp.zeros((hidden_size, input_size))  # PyTorch Linear stores (out_features, in_features)
        self.bias = jnp.zeros(hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # PyTorch Linear: y = x @ weight.T + bias
        # weight shape is (hidden_size, input_size), so we need to transpose
        x = jnp.matmul(x, self.matmul_weight.T) + self.matmul_bias
        x = x * self.scale_factor
        x = x + x
        x = jnp.clip(x, self.clamp_min, self.clamp_max)
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
        # Mish activation: x * tanh(softplus(x))
        # softplus(x) = log(1 + exp(x))
        softplus_x = jnp.logaddexp(x, 0.0)  # More numerically stable softplus
        mish_x = x * jnp.tanh(softplus_x)
        # The original code does: x = x * mish(x), not just mish(x)
        x = x * mish_x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]