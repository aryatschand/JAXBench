"""
JAXBench Level 1 - Task 84: conv_depthwise_2D_asymmetric_input_square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.146002
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.groups = in_channels
        
        # Initialize weights but they will be overwritten by set_weights()
        key = jax.random.PRNGKey(0)
        # For depthwise conv with feature_group_count=in_channels:
        # kernel shape should be (kH, kW, 1, in_channels) for HWIO format
        weight_shape = (kernel_size, kernel_size, 1, in_channels)
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.bias_param = jnp.zeros((out_channels,))
        else:
            self.bias_param = None

    def set_weights(self, weights_dict):
        # PyTorch depthwise weight shape: (in_channels, 1, kH, kW) 
        # where groups=in_channels means each input channel has its own filter
        # JAX expects (kH, kW, in_features/groups, out_features) = (kH, kW, 1, in_channels)
        w = weights_dict['conv2d.weight']
        # PyTorch: (C, 1, kH, kW) -> JAX: (kH, kW, 1, C)
        w = jnp.transpose(w, (2, 3, 1, 0))
        self.weight = jnp.array(w)
        
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Depthwise convolution using lax.conv_general_dilated
        # For depthwise conv: feature_group_count = in_channels
        # lhs (input): (N, H, W, C) where C = in_channels
        # rhs (kernel): (kH, kW, C/groups, C) = (kH, kW, 1, in_channels)
        out = lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups
        )
        
        if self.bias_param is not None:
            out = out + self.bias_param.reshape(1, 1, 1, -1)
            
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 64
in_channels = 128  
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]