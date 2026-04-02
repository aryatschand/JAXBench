import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        block = (8, 128)
        grid = (x.shape[0] // block[0], x.shape[1] // block[1])

        def kernel(x_ref, o_ref):
            x_val = x_ref[:, :]
            o_ref[:, :] = jnp.clip(x_val, -1.0, 1.0)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
