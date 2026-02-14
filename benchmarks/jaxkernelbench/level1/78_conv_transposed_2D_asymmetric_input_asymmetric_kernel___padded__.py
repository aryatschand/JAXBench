"""
JAXBench Level 1 - Task 78: conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.144810
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        # Initialize weights with same shapes as PyTorch
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        self.bias = None
        if bias:
            self.bias = jnp.zeros(out_channels)
            
        self.stride = stride
        self.kernel_size = kernel_size
        # For ConvTranspose2d, padding needs to be adjusted:
        # pad = kernel_size - 1 - pytorch_padding
        self.padding = ((kernel_size[0]-1-padding[0], kernel_size[0]-1-padding[0]),
                       (kernel_size[1]-1-padding[1], kernel_size[1]-1-padding[1]))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv_transpose2d.weight':
                # Convert PyTorch weight (in_channels, out_channels, kH, kW) to 
                # JAX format (kH, kW, out_channels, in_channels)
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
            elif name == 'conv_transpose2d.bias':
                value = jnp.array(value)
            setattr(self, name.replace('conv_transpose2d.', ''), jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Perform transposed convolution using lax.conv_transpose
        out = lax.conv_transpose(
            x,
            self.weight,
            strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, -1)

        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 32, 512, 1024))
    return [x]

def get_init_inputs():
    return [32, 32, (3, 7), (1, 1), (1, 3)]