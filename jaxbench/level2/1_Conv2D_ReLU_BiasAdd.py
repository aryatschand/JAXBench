"""
JAXBench Level 2 - Conv2D_ReLU_BiasAdd
Translated from KernelBench PyTorch to JAX using bedrock/opus.
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_shape = bias_shape
        
        # Initialize conv weights: (out_channels, in_channels, H, W)
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Kaiming-like initialization for conv weights
        fan_in = in_channels * kernel_size * kernel_size
        std = jnp.sqrt(2.0 / fan_in)
        self.conv_weight = jax.random.normal(key1, (out_channels, in_channels, kernel_size, kernel_size)) * std
        self.conv_bias = jax.random.normal(key2, (out_channels,)) * 0.01
        
        # Learnable bias parameter
        self.bias = jax.random.normal(key3, bias_shape)
    
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
    
    def forward(self, x):
        # Convolution using lax.conv_general_dilated
        # Input: (N, C, H, W), Kernel: (out_channels, in_channels, kH, kW)
        x = lax.conv_general_dilated(
            x,
            self.conv_weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        # Add conv bias
        x = x + self.conv_bias.reshape(1, -1, 1, 1)
        # ReLU activation
        x = jnp.maximum(x, 0)
        # Add learnable bias
        x = x + self.bias
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(42)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]