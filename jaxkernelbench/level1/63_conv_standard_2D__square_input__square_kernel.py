"""
JAXBench Level 1 - Task 63: conv_standard_2D__square_input__square_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    
    PyTorch Conv2d: kernel shape (out_channels, in_channels, H, W)
    JAX conv_general_dilated with NHWC: kernel shape (H, W, in_channels, out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.use_bias = bias
        
        k = kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (k, k, in_channels // groups, out_channels))
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
    
    def set_weights(self, weights_dict):
        """Set weights from PyTorch model."""
        for name, value in weights_dict.items():
            if 'weight' in name:
                # PyTorch Conv2d: (out_channels, in_channels, H, W)
                # JAX: (H, W, in_channels, out_channels)
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)
    
    def forward(self, x):
        """x: (batch, in_channels, H, W) - NCHW format"""
        # Convert to NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        
        ph, pw = self.padding
        
        out = lax.conv_general_dilated(
            x_nhwc,
            self.weight,
            window_strides=self.stride,
            padding=((ph, ph), (pw, pw)),
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups
        )
        
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        
        return out


# Test code
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

