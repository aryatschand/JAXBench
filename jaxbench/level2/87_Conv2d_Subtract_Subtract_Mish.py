"""
JAXBench Level 2 - Conv2d_Subtract_Subtract_Mish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        # Initialize with PyTorch conv2d weight shape (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Convert kernel (out,in,H,W) -> (H,W,in,out)
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
        
        # Convert back NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Subtract values
        x = x - self.subtract_value_1
        x = x - self.subtract_value_2
        
        # Mish activation: x * tanh(softplus(x))
        x = x * jnp.tanh(jnp.log1p(jnp.exp(x)))
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]