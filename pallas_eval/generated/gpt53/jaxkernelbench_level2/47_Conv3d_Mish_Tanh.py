"""
JAXBench Level 2 - Conv3d_Mish_Tanh (Pallas optimized)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def activation_kernel(x_ref, b_ref, o_ref):
    x = x_ref[...]
    b = b_ref[...]
    x = x + b
    x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
    x = jnp.tanh(x)
    o_ref[...] = x


def fused_activation(x, bias):
    N, D, H, W, C = x.shape
    x2d = x.reshape(N * D * H, W * C)
    b2d = jnp.broadcast_to(bias.reshape(1, C), (N * D * H, C))
    b2d = jnp.repeat(b2d, W, axis=1)

    block_m = 128
    block_n = 128

    grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

    out = pl.pallas_call(
        activation_kernel,
        out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        ),
    )(x2d, b2d)

    return out.reshape(N, D, H, W, C)


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else ((padding, padding), (padding, padding), (padding, padding))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        x = fused_activation(x, self.bias)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, D, H, W))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
