"""
JAXBench Level 2 - Conv2d_AvgPool_Sigmoid_Sum
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d: NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Transpose kernel: OIHW -> HWIO
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution
        x = jax.lax.conv_general_dilated(
            x, kernel, 
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, -1)

        # Average pooling
        pool_window = (1, self.pool_kernel_size, self.pool_kernel_size, 1)
        pool_strides = (1, self.pool_kernel_size, self.pool_kernel_size, 1)
        x = jax.lax.reduce_window(
            x, 0.0, jax.lax.add,
            window_dimensions=pool_window,
            window_strides=pool_strides,
            padding='VALID'
        )
        x = x / (self.pool_kernel_size * self.pool_kernel_size)

        # Back to NCHW format
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Sigmoid and sum
        x = jax.nn.sigmoid(x)
        x = jnp.sum(x, axis=(1,2,3))
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]