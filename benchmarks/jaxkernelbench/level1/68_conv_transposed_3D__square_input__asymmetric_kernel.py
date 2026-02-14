"""
JAXBench Level 1 - Task 68: conv_transposed_3D__square_input__asymmetric_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Performs a transposed 3D convolution with square input and asymmetric kernel.
    
    PyTorch ConvTranspose3d: kernel shape (in_channels, out_channels, D, H, W)
    JAX conv_transpose with NDHWC: kernel shape (D, H, W, out_channels, in_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.groups = groups
        self.use_bias = bias
        
        kd, kh, kw = self.kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (kd, kh, kw, out_channels // groups, in_channels // groups))
        
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
        
        # JAX conv_transpose padding: kernel_size - 1 - pytorch_padding
        pad_d = kd - 1 - pd
        pad_h = kh - 1 - ph
        pad_w = kw - 1 - pw
        
        out = lax.conv_transpose(
            x_ndhwc,
            self.weight,
            strides=self.stride,
            padding=((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        # Convert back to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)
        
        return out


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth, width, height))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]

