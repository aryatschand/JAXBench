import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a Sigmoid activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        def sigmoid_kernel(x_ref, o_ref):
            x_val = x_ref[:, :]
            o_ref[:, :] = 1.0 / (1.0 + jnp.exp(-x_val))

        # Ensure input is 2D (TPU requirement already satisfied here)
        M, N = x.shape

        block_m = 8
        block_n = 128

        grid = (M // block_m, N // block_n)

        return pl.pallas_call(
            sigmoid_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
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
