import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        N = A.shape[0]
        block_m = 128
        block_n = 128

        def matmul_kernel(a_ref, b_ref, o_ref):
            a = a_ref[...]        # (128, N)
            b = b_ref[...]        # (N, 128)
            o_ref[...] = jnp.matmul(a, b)

        grid_m = N // block_m
        grid_n = N // block_n

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((N, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, N), lambda i, j: (i, 0)),
                    pl.BlockSpec((N, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(A, B)

N = 2048 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, N))
    B = jax.random.uniform(key2, shape=(N, N))
    return [A, B]

def get_init_inputs():
    return []
