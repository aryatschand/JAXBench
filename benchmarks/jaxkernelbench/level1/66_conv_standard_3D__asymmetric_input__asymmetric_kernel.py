"""
JAXBench Level 1 - Task 66: conv_standard_3D__asymmetric_input__asymmetric_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    
    PyTorch Conv3d: kernel shape (out_channels, in_channels, D, H, W)
    JAX conv_general_dilated with NDHWC: kernel shape (D, H, W, in_channels, out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.groups = groups
        self.use_bias = bias
        
        kd, kh, kw = self.kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (kd, kh, kw, in_channels // groups, out_channels))
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
    
    def set_weights(self, weights_dict):
        """Set weights from PyTorch model."""
        for name, value in weights_dict.items():
            if 'weight' in name:
                # PyTorch Conv3d: (out_channels, in_channels, D, H, W)
                # JAX: (D, H, W, in_channels, out_channels)
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 4, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)
    
    def forward(self, x):
        """x: (batch, in_channels, D, H, W) - NCDHW format"""
        # Convert to NDHWC
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        pd, ph, pw = self.padding
        
        out = lax.conv_general_dilated(
            x_ndhwc,
            self.weight,
            window_strides=self.stride,
            padding=((pd, pd), (ph, ph), (pw, pw)),
            lhs_dilation=(1, 1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
            feature_group_count=self.groups
        )
        
        # Convert back to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)
        
        return out


# Test code
batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 128
width = 128


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

