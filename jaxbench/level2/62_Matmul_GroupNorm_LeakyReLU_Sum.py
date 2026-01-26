"""
JAXBench Level 2 - Matmul_GroupNorm_LeakyReLU_Sum
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        # Initialize weights with same shapes as PyTorch
        self.fc_weight = jnp.zeros((hidden_size, input_size))
        self.fc_bias = jnp.zeros(hidden_size)
        
        # Group norm parameters
        self.num_groups = num_groups
        self.num_channels = hidden_size
        self.eps = eps
        self.gn_weight = jnp.ones(hidden_size)
        self.gn_bias = jnp.zeros(hidden_size)
        
        self.negative_slope = negative_slope

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.fc_weight.T) + self.fc_bias

        # Group normalization
        N, C = x.shape
        group_size = C // self.num_groups
        x = x.reshape(N, self.num_groups, group_size)
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C)
        x = self.gn_weight * x + self.gn_bias

        # Leaky ReLU
        x = jnp.where(x > 0, x, x * self.negative_slope)

        # Element-wise sum
        x = x + x
        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]