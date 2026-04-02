import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        if self.dim != 1:
            return jnp.max(x, axis=self.dim)

        B, D1, D2 = x.shape

        block_b = 1
        block_k = 315  # 4095 / 315 = 13
        grid = (B, D2 // block_k)

        def kernel(x_ref, o_ref):
            acc = jnp.full((block_b, block_k), -jnp.inf, dtype=x_ref.dtype)

            def body(i, acc):
                slice_i = x_ref[:, i:i+1, :]
                slice_i = jnp.squeeze(slice_i, axis=1)
                return jnp.maximum(acc, slice_i)

            acc = jax.lax.fori_loop(0, D1, body, acc)
            o_ref[...] = acc

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, D2), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_b, 1, block_k), lambda i, j: (i, 0, j)),
                ],
                out_specs=pl.BlockSpec((block_b, block_k), lambda i, j: (i, j)),
            ),
        )(x)

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
