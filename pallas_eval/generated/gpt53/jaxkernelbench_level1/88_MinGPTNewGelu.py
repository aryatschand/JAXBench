"""
JAXBench Level 1 - Task 88: MinGPTNewGelu
Pallas TPU kernel implementation
"""

import jax
import jax.numpy as jnp
import math
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def gelu_kernel(x_ref, o_ref):
    x = x_ref[...]
    c = jnp.sqrt(2.0 / jnp.pi)
    y = 0.5 * x * (1.0 + jnp.tanh(c * (x + 0.044715 * x * x * x)))
    o_ref[...] = y


class Model:
    def __init__(self):
        pass

    def forward(self, x):
        M, N = x.shape

        block = (128, 128)
        grid = (M // block[0], N // block[1])

        return pl.pallas_call(
            gelu_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass


batch_size = 8192
dim = 8192


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, dim))]


def get_init_inputs():
    return []
