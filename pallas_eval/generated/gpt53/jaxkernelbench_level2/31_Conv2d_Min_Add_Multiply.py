"""
JAXBench Level 2 - Conv2d_Min_Add_Multiply (Pallas TPU optimized)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def post_kernel(x_ref, bias_ref, o_ref, constant_value, scaling_factor):
    x = x_ref[...]
    bias = bias_ref[...]

    x = jnp.minimum(x, constant_value)
    x = x + bias
    x = x * scaling_factor

    o_ref[...] = x


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        # HWIO kernel
        kernel = jnp.transpose(self.conv_weight, (2, 3, 1, 0))

        # Convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        # Add conv bias
        x = x + self.conv_bias

        # Back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape

        # Flatten spatial dims for 2D requirement
        x_2d = x.reshape(N * C, H * W)
        bias = self.bias.reshape(C, 1)
        bias = jnp.repeat(bias, N, axis=0)  # (N*C,1)
        bias_2d = jnp.broadcast_to(bias, (N * C, H * W))

        block_m = 128
        block_n = 128

        grid = (x_2d.shape[0] // block_m, x_2d.shape[1] // block_n)

        out = pl.pallas_call(
            lambda x_ref, b_ref, o_ref: post_kernel(
                x_ref, b_ref, o_ref, self.constant_value, self.scaling_factor
            ),
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_2d, bias_2d)

        return out.reshape(N, C, H, W)


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]
