"""
JAXBench Level 2 - Gemm_BiasAdd_Hardtanh_Mish_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)
        self.groupnorm_weight = jnp.ones(out_features)
        self.groupnorm_bias = jnp.zeros(out_features)
        self.num_groups = num_groups
        self.num_channels = out_features

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear
        x = jnp.matmul(x, self.weight)
        
        # Bias add
        x = x + self.bias

        # Hardtanh
        x = jnp.clip(x, -1.0, 1.0)

        # Mish
        x = x * jnp.tanh(jax.nn.softplus(x))

        # GroupNorm
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_groups, -1)
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(batch_size, -1)
        x = x * self.groupnorm_weight + self.groupnorm_bias

        return x


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]