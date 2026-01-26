"""
JAXBench Level 2 - Conv2d_Min_Add_Multiply
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        # Initialize weights with same shapes as PyTorch
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (out,in,H,W) to (H,W,in,out)
        kernel = jnp.transpose(self.conv_weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding (PyTorch default for Conv2d)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add conv bias (in NHWC format, bias is added to last dimension)
        x = x + self.conv_bias
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Min with constant
        x = jnp.minimum(x, self.constant_value)
        
        # Add bias (shape is (out_channels, 1, 1))
        x = x + self.bias
        
        # Scale
        x = x * self.scaling_factor
        
        return x

batch_size = 128
in_channels = 64  
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]