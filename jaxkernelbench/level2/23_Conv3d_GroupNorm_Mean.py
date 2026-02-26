"""
JAXBench Level 2 - Conv3d_GroupNorm_Mean
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        # Conv3d weights shape: (out_channels, in_channels, D, H, W)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        
        # GroupNorm parameters
        self.num_groups = num_groups
        self.gamma = jnp.ones(out_channels)
        self.beta = jnp.zeros(out_channels)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Conv3d
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        # Convert kernel (out,in,D,H,W) -> (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW for GroupNorm
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Group Normalization
        N, C, D, H, W = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G, D, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, D, H, W)
        x = x * self.gamma.reshape(1, -1, 1, 1, 1) + self.beta.reshape(1, -1, 1, 1, 1)
        
        # Mean across all dimensions except batch
        x = jnp.mean(x, axis=(1, 2, 3, 4))
        
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]