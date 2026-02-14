"""
JAXBench Level 1 - Task 59: conv_standard_3D__asymmetric_input__square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.140935
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights and bias
        kernel_shape = (kernel_size, kernel_size, 1, in_channels, out_channels)
        k = 1.0 / (in_channels * kernel_size * kernel_size)
        rng = jax.random.PRNGKey(0)
        self.weight = jax.random.uniform(rng, kernel_shape) * jnp.sqrt(k)
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv3d.weight':
                # Convert from PyTorch (out_channels, in_channels, kD, kH, kW) to 
                # JAX (kD, kH, kW, in_channels, out_channels)
                value = jnp.transpose(value, (2, 3, 4, 1, 0))
            elif name == 'conv3d.bias':
                value = jnp.array(value)
            setattr(self, name.replace('conv3d.', ''), value)

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Apply 3D convolution
        out = lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=self.stride,
            padding=[(p, p) for p in self.padding],
            lhs_dilation=(1, 1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
            feature_group_count=self.groups
        )
        
        if self.use_bias:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)
            
        # Convert back from NDHWC to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width, depth))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]