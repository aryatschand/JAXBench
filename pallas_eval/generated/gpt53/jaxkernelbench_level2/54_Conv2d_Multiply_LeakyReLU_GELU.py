"""
JAXBench Level 2 - Conv2d_Multiply_LeakyReLU_GELU
Pallas TPU optimized version
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, m_ref, o_ref):
    x = x_ref[:, :]
    m = m_ref[:, :]
    y = x * m
    y = jnp.where(y > 0, y, 0.01 * y)
    y = jnn.gelu(y)
    o_ref[:, :] = y


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, -1)

        # NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape

        # reshape to 2D
        x2 = x.reshape(N * C, H * W)

        # prepare multiplier per row
        mul = jnp.reshape(self.multiplier, (C,))
        mul = jnp.repeat(mul, N).reshape(N * C, 1)

        # tile to match width
        mul2 = jnp.broadcast_to(mul, x2.shape)

        block = (128, 128)
        grid = (x2.shape[0] // block[0], x2.shape[1] // block[1])

        y2 = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x2, mul2)

        y = y2.reshape(N, C, H, W)
        return y


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
