"""
JAXBench Level 1 - Task 69: conv_transposed_2D__asymmetric_input__asymmetric_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Performs a transposed 2D convolution with asymmetric input and kernel.
    
    PyTorch ConvTranspose2d: kernel shape (in_channels, out_channels, H, W)
    JAX conv_transpose with NHWC: kernel shape (H, W, out_channels, in_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.use_bias = bias
        
        kh, kw = self.kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (kh, kw, out_channels // groups, in_channels // groups))
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
    
    def set_weights(self, weights_dict):
        """Set weights from PyTorch model."""
        for name, value in weights_dict.items():
            if 'weight' in name:
                # PyTorch: (in_channels, out_channels/groups, H, W)
                # JAX: (H, W, out_channels/groups, in_channels/groups)
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)
    
    def forward(self, x):
        """x: (batch, in_channels, H, W) - NCHW format"""
        # Convert to NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        
        kh, kw = self.kernel_size
        ph, pw = self.padding
        
        # JAX conv_transpose padding: kernel_size - 1 - pytorch_padding
        pad_h = kh - 1 - ph
        pad_w = kw - 1 - pw
        
        out = lax.conv_transpose(
            x_nhwc,
            self.weight,
            strides=self.stride,
            padding=((pad_h, pad_h), (pad_w, pad_w)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            rhs_dilation=self.dilation
        )
        
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        
        return out


# Test code
batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
height_in = 128
width_in = 256


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, height_in, width_in))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
