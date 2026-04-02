"""
JAXBench Level 1 - Task 89: cumsum
Pallas TPU kernel implementation
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cumsum_kernel(x_ref, o_ref):
    x = x_ref[0, :]
    y = jnp.cumsum(x)
    o_ref[0, :] = y


class Model:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        # Only optimized for dim=1 (row-wise cumsum)
        if self.dim != 1:
            return jnp.cumsum(x, axis=self.dim)

        batch, width = x.shape

        return pl.pallas_call(
            cumsum_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch,),
                in_specs=[pl.BlockSpec((1, width), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((1, width), lambda i: (i, 0)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]
