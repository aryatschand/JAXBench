"""
JAXBench Level 2 - ConvTranspose2d_GELU_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 19: ConvTranspose2d_GELU_GroupNorm
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.num_groups = num_groups
        self.eps = 1e-5

        # ConvTranspose2d weights - PyTorch shape: (in_channels, out_channels, kH, kW)
        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))

        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, H, W) in PyTorch format
        # Convert to NHWC for JAX
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # ConvTranspose2d using manual approach with kernel flipping
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1))

        # For PyTorch ConvTranspose2d with padding=0:
        # JAX padding = kernel_size - 1 - 0 = kernel_size - 1
        k = self.kernel_size
        pad = k - 1
        jax_padding = ((pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding=jax_padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        # Add bias
        x = x + self.conv_transpose_bias.reshape(1, 1, 1, -1)

        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # GELU activation
        x = jax.nn.gelu(x)

        # GroupNorm (use biased variance like PyTorch)
        N, C, H, W = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        x = x * self.group_norm_weight.reshape(1, -1, 1, 1) + self.group_norm_bias.reshape(1, -1, 1, 1)

        return x

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
groups = 8
num_groups = 8

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
