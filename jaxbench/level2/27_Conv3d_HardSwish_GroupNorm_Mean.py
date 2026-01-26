"""
JAXBench Level 2 - Conv3d_HardSwish_GroupNorm_Mean
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import relu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        # Conv3d weights shape: (out_channels, in_channels, D, H, W)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = jnp.zeros(out_channels)
        
        # GroupNorm parameters
        self.num_groups = num_groups
        self.gamma = jnp.ones(out_channels)
        self.beta = jnp.zeros(out_channels)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Conv3d: NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
        
        if hasattr(self, 'bias'):
            x = x + self.bias.reshape(1, 1, 1, 1, -1)
            
        # Back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # HardSwish activation
        x = x * jnp.minimum(jnp.maximum(x + 3, 0), 6) / 6
        
        # GroupNorm
        N, C, D, H, W = x.shape
        x = x.reshape(N, self.num_groups, C // self.num_groups, D, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, D, H, W)
        x = self.gamma.reshape(1, -1, 1, 1, 1) * x + self.beta.reshape(1, -1, 1, 1, 1)
        
        # Mean over spatial dimensions
        x = jnp.mean(x, axis=(2, 3, 4))
        
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4