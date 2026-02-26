"""
JAXBench Level 2 - ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 11: ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_groups = num_groups
        self.eps = 1e-5

        # Conv transpose weights - PyTorch shape: (in_channels, out_channels, kH, kW)
        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))

        # BatchNorm parameters (learnable only - training mode uses batch statistics)
        self.batch_norm_weight = jnp.ones((out_channels,))
        self.batch_norm_bias = jnp.zeros((out_channels,))

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

        # ConvTranspose2d
        # PyTorch weight: (in_channels, out_channels, kH, kW)
        # JAX conv_transpose expects: (kH, kW, out_channels, in_channels)
        weight = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))

        # Correct padding for JAX conv_transpose to match PyTorch
        # For PyTorch ConvTranspose2d with padding=p:
        # JAX padding = kernel_size - 1 - pytorch_padding
        k = self.kernel_size
        pad = k - 1 - self.padding
        jax_padding = ((pad, pad), (pad, pad))

        x = lax.conv_transpose(
            x, weight,
            strides=(self.stride, self.stride),
            padding=jax_padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        x = x + self.conv_transpose_bias.reshape(1, 1, 1, -1)

        # Convert back to NCHW for BatchNorm
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # BatchNorm2d (training mode - use batch statistics)
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)
        bn_weight = self.batch_norm_weight.reshape(1, -1, 1, 1)
        bn_bias = self.batch_norm_bias.reshape(1, -1, 1, 1)
        x = (x - mean) / jnp.sqrt(var + self.eps) * bn_weight + bn_bias

        # Tanh
        x = jnp.tanh(x)

        # MaxPool2d with kernel_size=2, stride=2
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

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

batch_size = 512
in_channels = 64
out_channels = 128
height = width = 32
kernel_size = 5
stride = 1
padding = 1
groups = 8
num_groups = 8

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
