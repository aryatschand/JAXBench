import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a Swish activation.
    """
    def __init__(self):
        pass

    def forward(self, x):
        def kernel_fn(x_ref, o_ref):
            x_val = x_ref[...]
            o_ref[...] = x_val * jax.nn.sigmoid(x_val)

        # Choose TPU-friendly block sizes
        block_m = 128
        block_n = 1024

        grid_m = x.shape[0] // block_m
        grid_n = x.shape[1] // block_n

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
