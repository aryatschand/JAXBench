"""
JAXBench Level 2 - Conv2d_BatchNorm_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 73: Conv2d_BatchNorm_Scaling
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.eps = 1e-5

        # Conv2d parameters
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        self.conv_weight = jax.random.normal(key1, (out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jax.random.normal(key2, (out_channels,))

        # BatchNorm2d parameters (learnable only)
        self.bn_weight = jnp.ones((out_channels,))
        self.bn_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, H, W) in PyTorch format
        # Convert to NHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # Conv2d (valid padding - no padding)
        # PyTorch weight: (out_channels, in_channels, kH, kW)
        # JAX conv expects: (kH, kW, in_channels, out_channels)
        weight = jnp.transpose(self.conv_weight, (2, 3, 1, 0))

        x = lax.conv_general_dilated(
            x, weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        # Add bias
        x = x + self.conv_bias.reshape(1, 1, 1, -1)

        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # BatchNorm2d (training mode - use batch statistics)
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)
        weight = self.bn_weight.reshape(1, -1, 1, 1)
        bias = self.bn_bias.reshape(1, -1, 1, 1)
        x = (x - mean) / jnp.sqrt(var + self.eps) * weight + bias

        # Scaling
        x = x * self.scaling_factor

        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
