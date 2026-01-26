"""
JAXBench Level 2 - Gemm_GroupNorm_Min_BiasAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_features, in_features))
        self.bias_linear = jnp.zeros(out_features)
        
        # GroupNorm parameters
        self.num_groups = num_groups
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)
        self.eps = 1e-5
        
        # Final bias
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.weight.T) + self.bias_linear

        # Group Normalization
        N, C = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G)
        
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C)
        
        x = x * self.group_norm_weight + self.group_norm_bias
        
        # Min operation along dim=1 (features), keepdim=True
        # After linear, x has shape (N, C) = (1024, 8192)
        # PyTorch min with dim=1, keepdim=True gives shape (N, 1)
        x = jnp.min(x, axis=1, keepdims=True)
        # x now has shape (1024, 1)
        
        # PyTorch output shape is (1, 8192, 1024, 1)
        # This means we need to reshape to (1, out_features, batch_size, 1)
        # x shape is (batch_size, 1) = (1024, 1)
        # We need to broadcast/reshape to match bias_shape (1, 8192, 1, 1)
        # and get output (1, 8192, 1024, 1)
        
        # Reshape x to (1, 1, N, 1) for broadcasting with bias (1, C, 1, 1)
        x = x.reshape(1, 1, N, 1)
        
        # Add bias - bias has shape (1, out_features, 1, 1)
        # Result will broadcast to (1, out_features, N, 1)
        x = x + self.bias
        
        return x

batch_size = 1024
in_features = 8192  
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]