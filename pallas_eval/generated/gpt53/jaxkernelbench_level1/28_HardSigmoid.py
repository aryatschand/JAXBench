import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        def kernel(x_ref, o_ref):
            v = x_ref[...]
            y = (v + 3.0) / 6.0
            y = jnp.where(y < 0.0, 0.0, y)
            y = jnp.where(y > 1.0, 1.0, y)
            o_ref[...] = y

        # Ensure 2D (already is in this workload)
        n, m = x.shape

        block = (128, 128)
        grid = (n // block[0], m // block[1])

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
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
