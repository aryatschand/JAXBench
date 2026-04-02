import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        def tanh_kernel(x_ref, o_ref):
            o_ref[...] = jnp.tanh(x_ref[...])

        # Ensure 2D (already is, but keep general)
        x_2d = x
        M, N = x_2d.shape

        block_m = 128
        block_n = 1024

        grid_m = M // block_m
        grid_n = N // block_n

        return pl.pallas_call(
            tanh_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_2d)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
