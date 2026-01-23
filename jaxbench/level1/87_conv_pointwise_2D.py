"""
JAXBench Level 1 - Task 87: conv_pointwise_2D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.147241
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        # Initialize weights with same shape as PyTorch
        # For Conv2d, PyTorch shape is (out_channels, in_channels, kH, kW)
        # Need to transpose to (kH, kW, in_channels, out_channels) for JAX
        weight_shape = (1, 1, in_channels, out_channels)
        self.conv1d_weight = jnp.zeros(weight_shape)
        self.conv1d_bias = jnp.zeros(out_channels) if bias else None
        self.use_bias = bias
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Transpose from PyTorch (out,in,H,W) to JAX (H,W,in,out)
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Perform convolution using lax.conv_general_dilated
        out = lax.conv_general_dilated(
            x,
            self.conv1d_weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add bias if present
        if self.conv1d_bias is not None:
            out = out + self.conv1d_bias
            
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

# Test code - REDUCED SIZE for memory
batch_size = 4  # Reduced from 16
in_channels = 64  
out_channels = 128
width = 512  # Reduced from 1024
height = 512  # Reduced from 1024

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]