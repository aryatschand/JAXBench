"""
JAXBench Level 1 - Task 52: Argmin_over_a_dimension
Pallas TPU kernel implementation
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def argmin_kernel(x_ref, o_ref):
    x = x_ref[...]  # shape: (B, D1, D2)
    idx = jnp.argmin(x, axis=1)
    o_ref[...] = idx.astype(jnp.int32)


class Model:
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self.dim == 1, "This kernel is specialized for dim=1"

        B, D1, D2 = x.shape

        b0 = 8
        b2 = 128

        assert B % b0 == 0
        assert D2 % b2 == 0

        return pl.pallas_call(
            argmin_kernel,
            out_shape=jax.ShapeDtypeStruct((B, D2), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(B // b0, D2 // b2),
                in_specs=[
                    pl.BlockSpec((b0, D1, b2), lambda i, j: (i, 0, j)),
                ],
                out_specs=pl.BlockSpec((b0, b2), lambda i, j: (i, j)),
            ),
        )(x)


batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]


def get_init_inputs():
    return [dim]
