"""
JAXBench Level 2 - Conv2d_Subtract_Subtract_Mish (Pallas TPU optimized)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def mish_kernel(x_ref, o_ref, subtract1, subtract2):
    x = x_ref[...]
    x = x - subtract1
    x = x - subtract2
    sp = jnp.log1p(jnp.exp(x))
    o_ref[...] = x * jnp.tanh(sp)


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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

        n, c, h, w = x.shape
        x2d = jnp.reshape(x, (n * c, h * w))

        block = (128, 4)
        grid = (x2d.shape[0] // block[0], x2d.shape[1] // block[1])

        def kernel(x_ref, o_ref):
            mish_kernel(x_ref, o_ref, self.subtract_value_1, self.subtract_value_2)

        out2d = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x2d)

        out = jnp.reshape(out2d, (n, c, h, w))
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
