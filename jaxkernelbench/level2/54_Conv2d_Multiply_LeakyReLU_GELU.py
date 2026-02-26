"""
JAXBench Level 2 - Conv2d_Multiply_LeakyReLU_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        # Initialize conv weights with same shape as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d: NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Transpose kernel to HWIO format
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding (PyTorch default is no padding)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, -1)
        
        # Back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Multiply by learnable scalar
        x = x * self.multiplier
        
        # LeakyReLU
        x = jnn.leaky_relu(x)
        
        # GELU
        x = jnn.gelu(x)
        
        return x

batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]