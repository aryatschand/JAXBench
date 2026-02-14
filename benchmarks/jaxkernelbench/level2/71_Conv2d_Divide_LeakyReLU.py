"""
JAXBench Level 2 - Conv2d_Divide_LeakyReLU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import leaky_relu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel for JAX conv
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Perform convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, -1)
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        x = x / self.divisor
        x = leaky_relu(x, negative_slope=0.01)
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]