"""
JAXBench Level 1 - Task 80: conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.145072
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        
        # Initialize weights with same shape as PyTorch but transposed for JAX
        kernel_shape = (kernel_size[0], kernel_size[1], in_channels, out_channels)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias_param = jnp.zeros(out_channels)
            
    def set_weights(self, weights_dict):
        # Convert PyTorch weights (out_channels, in_channels, kH, kW) to JAX (kH, kW, in_channels, out_channels)
        weight = weights_dict['conv2d.weight']
        self.weight = jnp.transpose(jnp.array(weight), (2, 3, 1, 0))
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])
            
    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Calculate padding for JAX
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        padding = ((pad_h, pad_h), (pad_w, pad_w))
        
        # Apply convolution using lax.conv_general_dilated
        y = lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=padding,
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        if self.use_bias:
            y = y + self.bias_param[None, None, None, :]
            
        # Convert back from NHWC to NCHW
        y = jnp.transpose(y, (0, 3, 1, 2))
        return y

# Test parameters
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]