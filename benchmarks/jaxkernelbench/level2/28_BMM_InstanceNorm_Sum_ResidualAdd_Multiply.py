"""
JAXBench Level 2 - BMM_InstanceNorm_Sum_ResidualAdd_Multiply
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        self.in_features = in_features
        self.out_features = out_features
        # Linear layer weight shape: PyTorch uses (out_features, in_features)
        self.bmm_weight = jnp.zeros((out_features, in_features))
        self.bmm_bias = jnp.zeros((out_features,))
        
        # Instance norm parameters
        self.instance_norm_weight = jnp.ones((out_features,))
        self.instance_norm_bias = jnp.zeros((out_features,))
        self.eps = eps
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x, y):
        # Linear layer - PyTorch Linear stores weight as (out_features, in_features)
        # So we need to transpose it for x @ weight.T + bias
        x = x @ self.bmm_weight.T + self.bmm_bias
        
        # Instance norm - for InstanceNorm2d with input shape (N, C, 1, 1)
        # We normalize over the spatial dimensions (H, W) which are both 1
        # This means each sample's each channel is normalized independently
        # With H=W=1, the mean equals the value and variance is 0
        # So (x - mean) / sqrt(var + eps) = 0 / sqrt(eps)
        # Then multiply by weight and add bias
        
        # For a single spatial point, instance norm effectively becomes:
        # output = weight * 0 + bias = bias (when var=0)
        # But let's implement it properly:
        
        x = jnp.expand_dims(jnp.expand_dims(x, 2), 3)  # (N, C) -> (N, C, 1, 1)
        
        # For InstanceNorm2d, normalize over spatial dims (2, 3)
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        
        # Apply affine transformation
        x = x * jnp.reshape(self.instance_norm_weight, (1, -1, 1, 1)) + \
            jnp.reshape(self.instance_norm_bias, (1, -1, 1, 1))
        
        x = jnp.squeeze(jnp.squeeze(x, axis=3), axis=2)  # (N, C, 1, 1) -> (N, C)
        
        # Residual addition and multiplication
        x = x + y
        x = x * y
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    return [
        jax.random.uniform(key1, (batch_size, in_features), dtype=jnp.float32),
        jax.random.uniform(key2, (batch_size, out_features), dtype=jnp.float32)
    ]

def get_init_inputs():
    return [in_features, out_features]