import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def min_reduce_kernel(x_ref, o_ref):
    # x_ref shape: (1, 4096, 45)
    # Reduce over axis=1 (the 4096 dimension)
    vals = x_ref[0, :, :]  # (4096, 45)
    o_ref[0, :] = jnp.min(vals, axis=0)


class Model:
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        if self.dim != 1:
            return jnp.min(x, axis=self.dim)

        B, D1, D2 = x.shape

        block = (1, D1, 45)
        grid = (B, D2 // 45)

        return pl.pallas_call(
            min_reduce_kernel,
            out_shape=jax.ShapeDtypeStruct((B, D2), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, 0, j)),
                ],
                out_specs=pl.BlockSpec((1, 45), lambda i, j: (i, j)),
            ),
        )(x)


batch_size = 128
dim1 = 4096
dim2 = 4095


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]


def get_init_inputs():
    return [1]
