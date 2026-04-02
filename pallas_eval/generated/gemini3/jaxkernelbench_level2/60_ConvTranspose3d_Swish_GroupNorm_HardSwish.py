import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Model that performs a 3D transposed convolution, applies Swish activation,
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps
        self.use_bias = bias

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

        # ConvTranspose3d using optimized lax.conv_general_dilated
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        k = self.kernel_size
        pad = k - 1 - self.padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        # Use lhs_dilation to natively handle the transposed convolution stride
        # This avoids the massive memory overhead and scatter of manual dilation
        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1,
