"""
JAXBench Level 1 - Task 58: conv_transposed_3D__asymmetric_input__asymmetric_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    
    PyTorch ConvTranspose3d: kernel shape (in_channels, out_channels, D, H, W)
    JAX conv_transpose with NCDHW: kernel shape (D, H, W, out_channels, in_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weight: JAX conv_transpose expects (D, H, W, out_channels, in_channels)
        # for input format NCDHW
        kd, kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        key = jax.random.PRNGKey(0)
        # PyTorch init shape: (in_channels, out_channels/groups, kD, kH, kW)
        # We'll transpose when setting weights
        self.weight = jax.random.normal(key, (kd, kh, kw, out_channels // groups, in_channels // groups))
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
    
    def set_weights(self, weights_dict):
        """Set weights from PyTorch model."""
        for name, value in weights_dict.items():
            if 'weight' in name:
                # PyTorch ConvTranspose3d weight: (in_channels, out_channels/groups, D, H, W)
                # JAX conv_transpose expects: (D, H, W, out_channels/groups, in_channels/groups)
                # Transpose from (in, out, D, H, W) -> (D, H, W, out, in)
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 4, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)
    
    def forward(self, x):
        """
        x: (batch, in_channels, D, H, W) - NCDHW format
        """
        # Convert to NDHWC for JAX
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # (N, D, H, W, C)
        
        # Calculate padding for conv_transpose
        # PyTorch padding reduces output size, so for transpose we need: kernel_size - 1 - padding
        kd, kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,) * 3
        pd, ph, pw = self.padding if isinstance(self.padding, tuple) else (self.padding,) * 3
        
        # JAX conv_transpose padding
        pad_d = kd - 1 - pd
        pad_h = kh - 1 - ph
        pad_w = kw - 1 - pw
        
        # Perform transposed convolution
        out = lax.conv_transpose(
            x_ndhwc,
            self.weight,
            strides=self.stride if isinstance(self.stride, tuple) else (self.stride,) * 3,
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
out_channels = 16
kernel_size = (3, 5, 7)
depth_in = 16
height_in = 32
width_in = 64


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth_in, height_in, width_in))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

