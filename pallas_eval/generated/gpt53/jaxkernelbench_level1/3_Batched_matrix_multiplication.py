import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    a = a_ref[:, :]
    b = b_ref[:, :]
    o_ref[...] = jnp.matmul(a, b)


class Model:
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        pass
    
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        batch, m, k = A.shape
        _, _, n = B.shape

        bm = 128
        bn = 128

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((batch, m, n), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch, m // bm, n // bn),
                in_specs=[
                    pl.BlockSpec((bm, k), lambda b, i, j: (b, i, 0)),
                    pl.BlockSpec((k, bn), lambda b, i, j: (b, 0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda b, i, j: (b, i, j)),
            ),
        )(A, B)


batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(batch_size, m, k))
    B = jax.random.uniform(key2, shape=(batch_size, k, n))
    return [A, B]


def get_init_inputs():
    return []
