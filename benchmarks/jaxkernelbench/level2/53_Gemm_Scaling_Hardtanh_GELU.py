"""
JAXBench Level 2 - Gemm_Scaling_Hardtanh_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight) + self.bias
        x = x * self.scaling_factor
        x = jnp.clip(x, self.hardtanh_min, self.hardtanh_max)
        x = x * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))
        return x

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]