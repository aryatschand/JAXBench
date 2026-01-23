"""
JAXBench Level 1 - Task 56: conv_standard_2D__asymmetric_input__asymmetric_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.140303
"""

import jax
import jax.numpy as jnp
import jax.lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        # Initialize weights with same shape as PyTorch
        # PyTorch conv weight shape: (out_channels, in_channels // groups, kH, kW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        
        # Initialize with proper shape - will be overwritten by set_weights
        weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        self.conv2d_weight = jnp.zeros(weight_shape, dtype=jnp.float32)
        if bias:
            self.conv2d_bias = jnp.zeros((out_channels,), dtype=jnp.float32)
        else:
            self.conv2d_bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            # Replace dots with underscores for attribute names
            attr_name = name.replace('.', '_')
            setattr(self, attr_name, jnp.array(value, dtype=jnp.float32))

    def forward(self, x):
        # Ensure float32
        x = x.astype(jnp.float32)
        
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose weight from (out_channels, in_channels // groups, kH, kW) to (kH, kW, in_channels // groups, out_channels)
        weight = jnp.transpose(self.conv2d_weight, (2, 3, 1, 0))
        
        out = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=self.stride,
            padding=[(self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])],
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups
        )

        if self.conv2d_bias is not None:
            out = out + self.conv2d_bias.reshape(1, 1, 1, -1)

        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

# Test code
batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size, in_channels, height, width), dtype=jnp.float32)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]