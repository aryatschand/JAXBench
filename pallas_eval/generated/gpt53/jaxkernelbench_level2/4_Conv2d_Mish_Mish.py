"""
JAXBench Level 2 - Conv2d_Mish_Mish (Pallas optimized Mish)
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mish2_kernel(x_ref, o_ref):
    x = x_ref[...]
    y = x * jnp.tanh(jnn.softplus(x))
    y = y * jnp.tanh(jnn.softplus(y))
    o_ref[...] = y

def mish2_pallas(x):
    n, c, h, w = x.shape
    x2 = x.reshape(n * c, h * w)

    block = (128, 128)
    grid = (x2.shape[0] // block[0], x2.shape[1] // block[1])

    y2 = pl.pallas_call(
        mish2_kernel,
        out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x2)

    return y2.reshape(n, c, h, w)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
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
        x = jnp.transpose(x, (0, 3, 1, 2))

        x = mish2_pallas(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
