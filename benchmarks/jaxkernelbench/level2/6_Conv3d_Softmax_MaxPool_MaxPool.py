"""
JAXBench Level 2 - Conv3d_Softmax_MaxPool_MaxPool
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import softmax

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        # Conv3d weight shape: (out_channels, in_channels, kD, kH, kW)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv3d
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        # Convert kernel (out,in,D,H,W) -> (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW for softmax
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        x = softmax(x, axis=1)
        
        # MaxPool3d operations
        # Convert to NDHWC for pooling
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        for _ in range(2): # Two pooling operations
            x = jax.lax.reduce_window(
                x,
                init_value=-jnp.inf,
                computation=jax.lax.max,
                window_dimensions=(1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1),
                window_strides=(1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1),
                padding='VALID'
            )

        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]