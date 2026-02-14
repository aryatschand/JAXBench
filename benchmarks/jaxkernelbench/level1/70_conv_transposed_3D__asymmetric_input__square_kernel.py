"""
JAXBench Level 1 - Task 70: conv_transposed_3D__asymmetric_input__square_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Performs a transposed 3D convolution with asymmetric input and square kernel.
    
    PyTorch ConvTranspose3d: kernel shape (in_channels, out_channels, D, H, W)
    JAX conv_transpose with NDHWC: kernel shape (D, H, W, out_channels, in_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.use_bias = bias
        
        k = kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (k, k, k, out_channels // groups, in_channels // groups))
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
    
    def set_weights(self, weights_dict):
        """Set weights from PyTorch model."""
        for name, value in weights_dict.items():
            if 'weight' in name:
                # PyTorch: (in_channels, out_channels/groups, D, H, W)
                # JAX: (D, H, W, out_channels/groups, in_channels/groups)
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 4, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)
    
    def forward(self, x):
        """x: (batch, in_channels, D, H, W) - NCDHW format"""
        # Convert to NDHWC
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        kd, kh, kw = self.kernel_size
        pd, ph, pw = self.padding
        
        # JAX conv_transpose padding
        pad_d = kd - 1 - pd
        pad_h = kh - 1 - ph
        pad_w = kw - 1 - pw
        
        out = lax.conv_transpose(
            x_ndhwc,
            self.weight,
            strides=self.stride,
            padding=((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
            rhs_dilation=self.dilation
        )
        
        # Convert back to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)
        
        return out


# Test code
batch_size = 8
in_channels = 48
out_channels = 24
kernel_size = 3
depth = 96
height = 96
width = 96


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

