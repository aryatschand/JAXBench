import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        # assume 2D (batch, dim)
        n, m = x.shape

        def kernel_fn(x_ref, o_ref):
            row = x_ref[0, :]
            cumsum = jnp.cumsum(row, axis=0)
            zero = jnp.zeros((1,), dtype=row.dtype)
            exclusive = jnp.concatenate([zero, cumsum[:-1]], axis=0)
            o_ref[0, :] = exclusive

        block = (1, m)
        grid = (n, 1)

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, 0))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, 0)),
            ),
        )(x)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]
