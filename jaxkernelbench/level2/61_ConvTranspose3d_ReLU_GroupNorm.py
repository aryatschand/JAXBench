"""
JAXBench Level 2 - ConvTranspose3d_ReLU_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 61: ConvTranspose3d_ReLU_GroupNorm
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.use_bias = bias
        self.eps = 1e-5

        # ConvTranspose3d weights - PyTorch shape: (in_channels, out_channels, kD, kH, kW)
        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,)) if bias else None

        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, D, H, W) in PyTorch format
        # Convert to NDHWC for JAX
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC

        # ConvTranspose3d using manual approach matching level1/task77
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        # For PyTorch ConvTranspose3d with padding=0:
        # JAX padding = kernel_size - 1 - 0 = kernel_size - 1
        k = self.kernel_size
        pad = k - 1
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        if self.conv_transpose_bias is not None:
            x = x + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)

        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # ReLU activation
        x = jax.nn.relu(x)

        # GroupNorm (use biased variance like PyTorch)
        N, C, D, H, W = x.shape
        G = self.groups
        x = x.reshape(N, G, C // G, D, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C, D, H, W)
        x = x * self.group_norm_weight.reshape(1, -1, 1, 1, 1) + self.group_norm_bias.reshape(1, -1, 1, 1, 1)

        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
