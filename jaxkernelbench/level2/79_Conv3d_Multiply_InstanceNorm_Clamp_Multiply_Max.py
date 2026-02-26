"""
JAXBench Level 2 - Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        # Initialize conv weights with same shape as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)
        
        # Instance norm parameters
        self.instance_norm_weight = jnp.ones(out_channels)
        self.instance_norm_bias = jnp.zeros(out_channels)
        
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _instance_norm(self, x):
        # Calculate mean and var across spatial dimensions
        mean = jnp.mean(x, axis=(2,3,4), keepdims=True)
        var = jnp.var(x, axis=(2,3,4), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        
        # Apply scale and shift
        weight = self.instance_norm_weight.reshape(-1, 1, 1, 1)
        bias = self.instance_norm_bias.reshape(-1, 1, 1, 1)
        return x * weight + bias

    def forward(self, x):
        # Convert NCDHW -> NDHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Prepare conv kernel
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # First multiplication
        x = x * self.multiplier
        
        # Instance normalization
        x = self._instance_norm(x)
        
        # Clamp
        x = jnp.clip(x, self.clamp_min, self.clamp_max)
        
        # Second multiplication
        x = x * self.multiplier
        
        # Max along channel dimension
        x = jnp.max(x, axis=1)
        
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]