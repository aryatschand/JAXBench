"""
JAXBench Level 2 - Conv2d_Scaling_Min
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel for JAX conv
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution
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
        
        # Scale
        x = x * self.scale_factor
        
        # Min along channel dimension (dim=1)
        x = jnp.min(x, axis=1, keepdims=True)
        
        return x

batch_size = 64
in_channels = 64  
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]