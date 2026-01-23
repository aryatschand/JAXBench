"""
JAXBench Level 1 - Task 55: conv_standard_2D__asymmetric_input__square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.139857
"""

import jax
import jax.numpy as jnp
import jax.lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights with correct shape for JAX conv
        # PyTorch shape: (out_channels, in_channels, kernel_size, kernel_size)
        # JAX shape: (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = None
        if bias:
            self.bias_param = None
            
    def set_weights(self, weights_dict):
        # Convert PyTorch weights to JAX format
        weight = weights_dict['conv2d.weight']
        # Transpose from (out_channels, in_channels, H, W) to (H, W, in_channels, out_channels)
        self.weight = jnp.transpose(jnp.array(weight), (2, 3, 1, 0))
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])
            
    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        out = jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            lhs_dilation=(1, 1),
            rhs_dilation=(self.dilation, self.dilation),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups
        )
        
        if self.use_bias:
            out = out + self.bias_param.reshape(1, 1, 1, -1)
            
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 8
height = 512  
width = 1024
in_channels = 64
out_channels = 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]