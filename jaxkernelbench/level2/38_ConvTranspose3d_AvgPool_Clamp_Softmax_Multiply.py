"""
JAXBench Level 2 - ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        # Initialize ConvTranspose3d weight with PyTorch shape (in_channels, out_channels, k, k, k)
        self.conv_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.clamp_min = clamp_min 
        self.clamp_max = clamp_max
        self.scale = jnp.ones((1, out_channels, 1, 1, 1))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Average pooling using reduce_window
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        pool_window = (1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1)
        pool_strides = (1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1)
        x = jax.lax.reduce_window(x, 0.0, jax.lax.add, pool_window, pool_strides, 'VALID')
        x = x / (self.pool_kernel_size ** 3)

        # ConvTranspose3d
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        padding = [(self.kernel_size - 1 - self.padding) for _ in range(3)]
        padding = [(p, p+self.output_padding) for p in padding]
        x = jax.lax.conv_transpose(x, kernel, (self.stride, self.stride, self.stride),
                                 padding,
                                 dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
        
        # Add bias
        x = x + self.conv_bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # Clamp
        x = jnp.clip(x, self.clamp_min, self.clamp_max)

        # Spatial softmax
        b, c, d, h, w = x.shape
        x = x.reshape(b, c, -1)
        x = jax.nn.softmax(x, axis=2)
        x = x.reshape(b, c, d, h, w)

        # Scale
        x = x * self.scale
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]