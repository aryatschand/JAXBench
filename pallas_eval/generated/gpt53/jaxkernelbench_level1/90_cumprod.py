"""
JAXBench Level 1 - Task 90: cumprod
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.148274
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cumprod_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    o_ref[:, :] = jnp.cumprod(x, axis=1)


class Model:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        assert self.dim == 1, "This optimized kernel only supports dim=1"

        # Ensure 2D
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        B, N = x.shape

        block = (8, N)
        grid = (B // block[0],)

        return pl.pallas_call(
            cumprod_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i: (i, 0))],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass


# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]
