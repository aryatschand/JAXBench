"""
JAXBench Level 1 - Task 38: L1Norm_
Pallas TPU kernel version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def l1_norm_kernel(x_ref, o_ref):
    x = x_ref[0, :]
    mean = jnp.mean(jnp.abs(x))
    o_ref[0, :] = x / mean


class Model:
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        pass

    def forward(self, x):
        batch, dim = x.shape

        return pl.pallas_call(
            l1_norm_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch,),
                in_specs=[
                    pl.BlockSpec((1, dim), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((1, dim), lambda i: (i, 0)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass


batch_size = 4096
dim = 8192


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []
