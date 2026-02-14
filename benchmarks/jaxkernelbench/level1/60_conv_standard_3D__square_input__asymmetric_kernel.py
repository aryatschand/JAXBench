"""
JAXBench Level 1 - Task 60: conv_standard_3D__square_input__asymmetric_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.141141
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        # Initialize with same shapes as PyTorch but transposed for JAX
        rng = jax.random.PRNGKey(0)
        weight_shape = (out_channels, in_channels, *kernel_size) # PyTorch shape
        k1, k2 = jax.random.split(rng)
        
        # Initialize weights similar to PyTorch default initialization
        weight = jax.random.normal(k1, weight_shape) * (1.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])) ** 0.5
        
        # Transpose weight from PyTorch (out,in,d,h,w) to JAX (d,h,w,in,out)
        self.weight = jnp.transpose(weight, (2,3,4,1,0))
        
        if bias:
            self.bias = jax.random.normal(k2, (out_channels,))
        else:
            self.bias = None
            
        self.stride = (stride, stride, stride)
        self.padding = [(padding, padding), (padding, padding), (padding, padding)]
        self.dilation = (dilation, dilation, dilation)
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv3d.weight':
                # Transpose from PyTorch (out,in,d,h,w) to JAX (d,h,w,in,out)
                value = jnp.transpose(jnp.array(value), (2,3,4,1,0))
                setattr(self, 'weight', value)
            elif name == 'conv3d.bias':
                setattr(self, 'bias', jnp.array(value))
            else:
                setattr(self, name.replace('conv3d.', '').replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0,2,3,4,1))
        
        out = lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=None,
            rhs_dilation=self.dilation,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
            feature_group_count=self.groups
        )

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)
            
        # Convert back from NDHWC to NCDHW
        out = jnp.transpose(out, (0,4,1,2,3))
        return out

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, width, height, depth))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]