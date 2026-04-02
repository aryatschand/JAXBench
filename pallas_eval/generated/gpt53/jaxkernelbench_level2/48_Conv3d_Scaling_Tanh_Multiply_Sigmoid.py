"""
JAXBench Level 2 - Conv3d_Scaling_Tanh_Multiply_Sigmoid
Pallas TPU optimized version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, scale_ref, bias_ref, o_ref):
    pid0 = pl.program_id(axis=0)
    pid1 = pl.program_id(axis=1)

    x = x_ref[...]

    # infer channel index from row
    rows_per_channel = x.shape[0]
    # global row offset
    row_offset = pid0 * rows_per_channel
    channel_idx = row_offset % scale_ref.shape[0]

    scale = scale_ref[channel_idx, 0, 0, 0]
    bias = bias_ref[channel_idx, 0, 0, 0]

    y = x * scale
    y = jnp.tanh(y)
    y = y * bias
    y = 1.0 / (1.0 + jnp.exp(-y))

    o_ref[...] = y


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias_conv = jnp.zeros(out_channels)
        self.scaling_factor = jnp.zeros(bias_shape)
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        x = x + self.bias_conv.reshape(1, 1, 1, 1, -1)

        # back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        N, C, D, H, W = x.shape

        # flatten to 2D for Pallas
        x_flat = x.reshape(N * C, D * H * W)

        block_m = 128
        block_n = 128

        grid = (
            x_flat.shape[0] // block_m,
            x_flat.shape[1] // block_n,
        )

        out = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((C, 1, 1, 1), lambda i, j: (0, 0, 0, 0)),
                    pl.BlockSpec((C, 1, 1, 1), lambda i, j: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_flat, self.scaling_factor, self.bias)

        return out.reshape(N, C, D, H, W)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
