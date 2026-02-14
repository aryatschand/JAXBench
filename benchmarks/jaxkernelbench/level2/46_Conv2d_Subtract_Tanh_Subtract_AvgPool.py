"""
JAXBench Level 2 - Conv2d_Subtract_Tanh_Subtract_AvgPool
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        # Initialize conv weights with same shape as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d: NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Transpose kernel: (out,in,H,W) -> (H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding (PyTorch default is no padding)
        x = jax.lax.conv_general_dilated(
            x, kernel, 
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, -1)
        
        # Back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Subtractions and activation
        x = x - self.subtract1_value
        x = jnp.tanh(x)
        x = x - self.subtract2_value
        
        # Average pooling
        x = jax.lax.reduce_window(
            x,
            init_value=0.,
            computation=jax.lax.add,
            window_dimensions=(1, 1, self.kernel_size_pool, self.kernel_size_pool),
            window_strides=(1, 1, self.kernel_size_pool, self.kernel_size_pool),
            padding='VALID'
        ) / (self.kernel_size_pool * self.kernel_size_pool)
        
        return x

batch_size = 128
in_channels = 64  
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]