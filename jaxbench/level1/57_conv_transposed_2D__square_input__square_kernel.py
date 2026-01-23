"""
JAXBench Level 1 - Task 57: conv_transposed_2D__square_input__square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.140631
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        # Initialize with shapes matching PyTorch weights
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.kernel_shape = kernel_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize dummy weights - will be replaced by set_weights()
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, kernel_shape)
        if bias:
            self.bias_param = jax.random.normal(key, (out_channels,))
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Store the raw PyTorch weight, we'll transform it in forward
                self.weight = jnp.array(value)
            elif 'bias' in name:
                self.bias_param = jnp.array(value)

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
        # For JAX conv_transpose with HWIO format: (kH, kW, out_channels, in_channels)
        # But with transpose_kernel=True, we use HWOI: (kH, kW, out_channels, in_channels)
        weight = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # For transposed convolution, the padding calculation is different
        # PyTorch output size: (input - 1) * stride - 2 * padding + kernel_size + output_padding
        # JAX conv_transpose with padding='SAME' or explicit padding
        
        # Use 'SAME' padding approach and then adjust
        # Actually, let's compute the correct padding for JAX
        # For conv_transpose, padding in JAX is the "effective" padding on the output
        
        # Perform transposed convolution using lax.conv_transpose
        # With transpose_kernel=False, the kernel is used as-is
        # The dimension_numbers specify the layout
        
        y = lax.conv_transpose(
            x,
            weight,
            strides=(self.stride, self.stride),
            padding=((self.kernel_size - 1 - self.padding, self.kernel_size - 1 - self.padding + self.output_padding),
                     (self.kernel_size - 1 - self.padding, self.kernel_size - 1 - self.padding + self.output_padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=False
        )
        
        if self.use_bias:
            y = y + self.bias_param.reshape(1, 1, 1, -1)
            
        # Convert back from NHWC to NCHW
        y = jnp.transpose(y, (0, 3, 1, 2))
        return y

# Test code
batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = 3
height = 1024
width = 1024

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]