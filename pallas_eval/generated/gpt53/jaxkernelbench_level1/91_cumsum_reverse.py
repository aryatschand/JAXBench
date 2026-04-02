import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        assert self.dim == 1, "This kernel is specialized for dim=1"

        def kernel(x_ref, o_ref):
            block = x_ref[:, :]
            out = jnp.flip(jnp.cumsum(jnp.flip(block, axis=1), axis=1), axis=1)
            o_ref[:, :] = out

        block_shape = (8, x.shape[1])
        grid = (x.shape[0] // block_shape[0], 1)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, 0))],
                out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, 0)),
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
