import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def argmax_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    idx = jnp.argmax(x, axis=1)
    o_ref[:, 0] = idx


class Model:
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        if self.dim != 1:
            return jnp.argmax(x, axis=self.dim)

        b, d1, d2 = x.shape
        x_reshaped = jnp.transpose(x, (0, 2, 1)).reshape(b * d2, d1)

        block_rows = 128
        block = (block_rows, d1)

        n_rows = x_reshaped.shape[0]
        grid = (n_rows // block_rows,)

        out = pl.pallas_call(
            argmax_kernel,
            out_shape=jax.ShapeDtypeStruct((n_rows, 1), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_rows, 1), lambda i: (i, 0)),
            ),
        )(x_reshaped)

        out = out.reshape(b, d2)
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
