"""
JAXBench Level 1 - Task 76: conv_standard_1D_dilated_strided__
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.144533
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = bias
        
        # Initialize weights with same shapes as PyTorch
        key = jax.random.PRNGKey(0)
        weight_shape = (kernel_size, in_channels, out_channels)
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv1d.weight':
                # Convert PyTorch weight (out_channels, in_channels, kernel_size) 
                # to JAX (kernel_size, in_channels, out_channels)
                value = jnp.transpose(jnp.array(value), (2, 1, 0))
                setattr(self, 'weight', value)
            elif name == 'conv1d.bias':
                value = jnp.array(value)
                setattr(self, 'bias', value)

    def forward(self, x):
        # Convert from PyTorch NCL to JAX NLC format
        x = jnp.transpose(x, (0, 2, 1))
        
        # Perform 1D convolution using lax.conv_general_dilated
        out = lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride,),
            padding='VALID',
            lhs_dilation=None,
            rhs_dilation=(self.dilation,),
            dimension_numbers=('NWC', 'WIO', 'NWC'),
            feature_group_count=1,
            batch_group_count=1,
            precision=None
        )
        
        if self.bias is not None:
            out = out + self.bias[None, None, :]
            
        # Convert back to PyTorch NCL format
        out = jnp.transpose(out, (0, 2, 1))
        return out

# Test code - REDUCED SIZE for memory
batch_size = 16  # Reduced from 64
in_channels = 64  
out_channels = 128
kernel_size = 3
length = 65536  # Reduced from 524280
stride = 3
dilation = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]