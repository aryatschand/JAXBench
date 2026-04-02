import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        pass

    def forward(self, A, B):
        N, M = B.shape

        A2 = jnp.reshape(A, (N, 1))

        block_n = 128
        block_m = 128

        def kernel(a_ref, b_ref, o_ref):
            a = a_ref[:, :]              # (block_n, 1)
            b = b_ref[:, :]              # (block_n, block_m)
            a_broadcast = pltpu.repeat(a, (1, b.shape[1]))
            o_ref[:, :] = b * a_broadcast

        grid = (N // block_n, M // block_m)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, M), B.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_n, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_n, block_m), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_n, block_m), lambda i, j: (i, j)),
            ),
        )(A2, B)


M = 4096
N = 4096


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N,))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]


def get_init_inputs():
    return []
