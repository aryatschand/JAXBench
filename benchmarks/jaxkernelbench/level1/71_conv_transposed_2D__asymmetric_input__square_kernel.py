"""
JAXBench Level 1 - Task 71: conv_transposed_2D__asymmetric_input__square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.143430
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights with zeros - will be set by set_weights()
        kernel_shape = (kernel_size, kernel_size, out_channels, in_channels)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias_param = jnp.zeros(out_channels)
        else:
            self.bias_param = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Convert from PyTorch (in_channels, out_channels, kH, kW) to JAX (kH, kW, out_channels, in_channels)
                value = np.transpose(value, (2, 3, 1, 0))
                self.weight = jnp.array(value)
            elif 'bias' in name:
                self.bias_param = jnp.array(value)
            
    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Calculate padding for transposed convolution
        pad = self.kernel_size - 1 - self.padding
        padding = ((pad, pad + self.output_padding), (pad, pad + self.output_padding))
        
        # Perform transposed convolution using lax.conv_transpose
        out = lax.conv_transpose(
            x,
            self.weight,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )
        
        if self.bias_param is not None:
            out = out + self.bias_param.reshape(1, 1, 1, -1)
            
        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

# Test code
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = 3
# large asymmetric input
height_in = 512
width_in = 1024

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]