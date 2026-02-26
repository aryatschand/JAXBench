"""
JAXBench Level 1 - Task 82: conv_depthwise_2D_square_input_square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.145345
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        
        # Initialize weights with same shape as PyTorch but transposed for JAX
        weight_shape = (in_channels, 1, kernel_size, kernel_size) # PyTorch shape
        key = jax.random.PRNGKey(0)
        weight = jax.random.normal(key, weight_shape) * 0.02
        # Transpose from (C_out, C_in, H, W) to (H, W, C_in, C_out) for JAX
        self.conv2d_weight = jnp.transpose(weight, (2, 3, 1, 0))
        
        if bias:
            self.conv2d_bias = jnp.zeros(in_channels)
        else:
            self.conv2d_bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Transpose weight from PyTorch to JAX format
                value = jnp.transpose(value, (2, 3, 1, 0))
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Apply depthwise convolution
        out = lax.conv_general_dilated(
            x,
            self.conv2d_weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.in_channels
        )
        
        if self.conv2d_bias is not None:
            out = out + self.conv2d_bias
            
        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]