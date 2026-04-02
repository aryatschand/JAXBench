import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])

        M, N = x2.shape
        bm, bn = 128, 128

        assert M % bm == 0 and N % bn == 0

        def sumsq_kernel(x_ref, o_ref):
            val = x_ref[...]
            o_ref[...] = jnp.sum(val * val)

        grid = (M // bm, N // bn)

        partials = pl.pallas_call(
            sumsq_kernel,
            out_shape=jax.ShapeDtypeStruct((grid[0], grid[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            ),
        )(x2)

        norm = jnp.sqrt(jnp.sum(partials))

        def div_kernel(x_ref, o_ref):
            o_ref[...] = x_ref[...] / norm

        out = pl.pallas_call(
            div_kernel,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x2)

        return out.reshape(orig_shape)

    def set_weights(self, weights_dict):
        pass

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return []
