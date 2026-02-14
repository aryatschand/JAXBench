"""
JAXBench Level 2 - Conv2d_ReLU_HardSwish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        # Initialize with PyTorch conv2d weight shape (out_channels, in_channels, kH, kW)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (out,in,H,W) to (H,W,in,out)
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
        
        # ReLU
        x = jnp.maximum(0, x)
        
        # HardSwish
        x = x * jnp.clip((x + 3) / 6, 0, 1)
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]