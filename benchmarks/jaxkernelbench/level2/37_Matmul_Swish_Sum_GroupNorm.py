"""
JAXBench Level 2 - Matmul_Swish_Sum_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)
        self.num_groups = num_groups
        self.out_features = out_features

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.weight)
        
        # Swish activation
        x = jax.nn.sigmoid(x) * x
        
        # Add bias
        x = x + self.bias

        # Group Norm
        group_size = self.out_features // self.num_groups
        x = x.reshape(-1, self.num_groups, group_size)
        
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        
        x = x.reshape(-1, self.out_features)
        x = x * self.group_norm_weight + self.group_norm_bias
        
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]