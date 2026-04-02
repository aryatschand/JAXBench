"""
JAXBench Level 2 - Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd
Pallas-optimized version
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_activation_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    b = bias_ref[...]

    # relu
    x = jnp.maximum(x, 0)

    # leaky relu
    x = jnp.where(x >= 0, x, 0.01 * x)

    # gelu
    x = gelu(x)

    # sigmoid
    x = 1 / (1 + jnp.exp(-x))

    # bias add
    o_ref[...] = x + b


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        x = x + self.conv_bias.reshape(1, 1, 1, 1, -1)

        # back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # reshape to 2D for Pallas: (N*C, D*H*W)
        N, C, D, H, W = x.shape
        x2 = x.reshape(N * C, D * H * W)

        # broadcast bias to match
        bias = self.bias.reshape(C, 1, 1, 1)
        bias = jnp.broadcast_to(bias, (C, D, H, W))
        bias2 = jnp.tile(bias.reshape(C, -1), (N, 1))

        block_m = 128
        block_n = 128

        m = x2.shape[0]
        n = x2.shape[1]

        bm = min(block_m, m)
        bn = min(block_n, n)

        grid = (m // bm, n // bn)

        out2 = pl.pallas_call(
            fused_activation_kernel,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x2, bias2)

        return out2.reshape(N, C, D, H, W)


batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
