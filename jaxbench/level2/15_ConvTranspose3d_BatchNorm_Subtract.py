"""
JAXBench Level 2 - ConvTranspose3d_BatchNorm_Subtract
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 15: ConvTranspose3d_BatchNorm_Subtract
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.eps = 1e-5

        # ConvTranspose3d weights - PyTorch shape: (in_channels, out_channels, kD, kH, kW)
        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,)) if bias else None

        # BatchNorm3d parameters (learnable only - training mode uses batch statistics)
        self.batch_norm_weight = jnp.ones((out_channels,))
        self.batch_norm_bias = jnp.zeros((out_channels,))

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
        # PyTorch weight: (in_channels, out_channels, kD, kH, kW)
        # JAX kernel: (kD, kH, kW, out_channels, in_channels)
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))

        # Flip the kernel for transposed convolution
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        # Dilate the input by inserting zeros for stride > 1
        batch_size, d_in, h_in, w_in, channels = x.shape
        k = self.kernel_size

        if self.stride > 1:
            d_dilated = d_in + (d_in - 1) * (self.stride - 1)
            h_dilated = h_in + (h_in - 1) * (self.stride - 1)
            w_dilated = w_in + (w_in - 1) * (self.stride - 1)

            x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
            x_dilated = x_dilated.at[:, ::self.stride, ::self.stride, ::self.stride, :].set(x)
            x = x_dilated

        # Padding formula: kernel_size - 1 - pytorch_padding
        pad = k - 1 - self.padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        if self.conv_transpose_bias is not None:
            x = x + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)

        # Convert back to NCDHW for BatchNorm
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # BatchNorm3d (training mode - use batch statistics)
        mean = jnp.mean(x, axis=(0, 2, 3, 4), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3, 4), keepdims=True)
        bn_weight = self.batch_norm_weight.reshape(1, -1, 1, 1, 1)
        bn_bias = self.batch_norm_bias.reshape(1, -1, 1, 1, 1)
        x = (x - mean) / jnp.sqrt(var + self.eps) * bn_weight + bn_bias

        # Subtract mean along spatial dimensions (D, H, W)
        spatial_mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        x = x - spatial_mean

        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
