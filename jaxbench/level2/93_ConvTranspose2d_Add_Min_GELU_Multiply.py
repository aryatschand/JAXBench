"""
JAXBench Level 2 - ConvTranspose2d_Add_Min_GELU_Multiply
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        # Initialize with PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, k, k)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.add_value = add_value
        self.multiply_value = multiply_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel (in,out,H,W) -> (H,W,out,in) for JAX conv_transpose
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Calculate padding (kernel_size - 1 - pytorch_padding)
        padding = ((self.kernel_size - 1, self.kernel_size - 1),
                  (self.kernel_size - 1, self.kernel_size - 1))
        
        # Transposed convolution
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, -1)
        
        # Convert back NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        x = x + self.add_value
        x = jnp.minimum(x, 0.0)
        x = gelu(x)
        x = x * self.multiply_value
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]