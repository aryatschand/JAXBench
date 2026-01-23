"""
JAXBench Level 1 - Task 67: conv_standard_1D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.143001
"""

import jax
import jax.numpy as jnp
from jax import lax

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
        
        # Initialize weights with same shapes as PyTorch
        kernel_shape = (out_channels, in_channels // groups, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias_param = jnp.zeros(out_channels)
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert NCL -> NLC for JAX conv
        x = jnp.transpose(x, (0, 2, 1))
        
        # Convert weight from (out_channels, in_channels/groups, kernel) to (kernel, in_channels/groups, out_channels)
        weight = jnp.transpose(self.weight, (2, 1, 0))
        
        # Apply 1D convolution using lax.conv_general_dilated
        y = lax.conv_general_dilated(
            x,
            weight,
            window_strides=(self.stride,),
            padding=[(self.padding, self.padding)],
            rhs_dilation=(self.dilation,),
            dimension_numbers=('NWC', 'WIO', 'NWC'),
            feature_group_count=self.groups
        )
        
        # Add bias if needed
        if self.use_bias:
            y = y + self.bias_param
            
        # Convert back NLC -> NCL
        y = jnp.transpose(y, (0, 2, 1))
        
        return y

def get_inputs():
    key = jax.random.PRNGKey(0)
    batch_size = 32
    in_channels = 64
    length = 131072
    x = jax.random.uniform(key, shape=(batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    in_channels = 64
    out_channels = 128 
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]