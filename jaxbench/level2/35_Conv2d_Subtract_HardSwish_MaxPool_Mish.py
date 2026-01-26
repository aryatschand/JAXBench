"""
JAXBench Level 2 - Conv2d_Subtract_HardSwish_MaxPool_Mish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d: NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution
        x = jax.lax.conv_general_dilated(
            x, kernel, 
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, -1)
        
        # Back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Subtract value
        x = x - self.subtract_value
        
        # HardSwish
        x = x * jnp.minimum(jnp.maximum(x + 3, 0), 6) / 6
        
        # MaxPool2d
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, self.pool_kernel_size, self.pool_kernel_size, 1),
            window_strides=(1, self.pool_kernel_size, self.pool_kernel_size, 1),
            padding='VALID'
        )
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        
        # Mish activation
        x = x * jnp.tanh(jax.nn.softplus(x))
        
        return x

batch_size = 128
in_channels = 64  
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]