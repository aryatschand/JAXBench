"""
JAXBench Level 2 - ConvTranspose3d_Max_Max_Sum
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # ConvTranspose3d weight shape: (in_channels, out_channels, D, H, W)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        pad_size = self.kernel_size - 1 - self.padding
        padding = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # MaxPool3d with kernel_size=2
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        )

        # MaxPool3d with kernel_size=3
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 3, 3, 3, 1),
            window_strides=(1, 3, 3, 3, 1),
            padding='VALID'
        )
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # Sum across channels
        x = jnp.sum(x, axis=1, keepdims=True)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]