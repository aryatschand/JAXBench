"""
JAXBench Level 2 - Gemm_GroupNorm_Hardtanh
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Initialize weights and bias for linear layer
        self.weight = jnp.zeros((out_features, in_features))
        self.bias = jnp.zeros(out_features)
        
        # Initialize group norm parameters
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.weight.T) + self.bias

        # Group Normalization
        # Reshape for group norm
        N, C = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G)
        
        # Calculate mean and var across spatial dims
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        
        # Reshape back and apply scale and bias
        x = x.reshape(N, C)
        x = x * self.group_norm_weight + self.group_norm_bias
        
        # HardTanh
        x = jnp.clip(x, self.hardtanh_min, self.hardtanh_max)
        
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]