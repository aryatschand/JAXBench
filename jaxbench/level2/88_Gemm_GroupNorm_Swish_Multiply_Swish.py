"""
JAXBench Level 2 - Gemm_GroupNorm_Swish_Multiply_Swish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        # Initialize weights with same shapes as PyTorch
        self.gemm_weight = jnp.zeros((out_features, in_features))
        self.gemm_bias = jnp.zeros((out_features,))
        
        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_features,))
        self.group_norm_bias = jnp.zeros((out_features,))
        self.num_groups = num_groups
        self.out_features = out_features
        
        # Multiply weight
        self.multiply_weight = jnp.zeros(multiply_weight_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.gemm_weight.T) + self.gemm_bias

        # GroupNorm - need to handle 2D input (batch_size, features)
        # PyTorch GroupNorm expects (N, C, *) but also works with (N, C)
        # We need to reshape to (N, num_groups, group_size) for normalization
        batch_size = x.shape[0]
        group_size = self.out_features // self.num_groups
        
        # Reshape to (batch_size, num_groups, group_size)
        x_grouped = x.reshape(batch_size, self.num_groups, group_size)
        
        # Compute mean and variance over the last axis (group_size)
        mean = jnp.mean(x_grouped, axis=-1, keepdims=True)
        var = jnp.var(x_grouped, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x_grouped - mean) / jnp.sqrt(var + 1e-5)
        
        # Reshape back to (batch_size, out_features)
        x = x_normalized.reshape(batch_size, self.out_features)
        
        # Apply affine transformation (weight and bias are per-channel)
        x = x * self.group_norm_weight + self.group_norm_bias

        # Swish
        x = x * jax.nn.sigmoid(x)
        
        # Multiply
        x = x * self.multiply_weight
        
        # Swish
        x = x * jax.nn.sigmoid(x)
        
        return x

batch_size = 1024
in_features = 8192  
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]