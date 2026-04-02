import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        # Only optimized for dim=1 as used in this workload
        assert self.dim == 1

        B, D1, D2 = x.shape

        def kernel(x_ref, o_ref):
            x_block = x_ref[...]            # (B, D1, 1)
            summed = jnp.sum(x_block, axis=1)  # (B, 1)
            o_ref[...] = summed / D1

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, D2), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1, D2),
                in_specs=[
                    pl.BlockSpec((B, D1, 1), lambda i, j: (i, 0, j)),
                ],
                out_specs=pl.BlockSpec((B, 1), lambda i, j: (i, j)),
            ),
        )(x)

        return out

    def set_weights(self, weights_dict):
        pass

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1]
