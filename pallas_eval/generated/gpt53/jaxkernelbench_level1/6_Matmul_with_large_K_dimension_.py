import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    a = a_ref[:, :]
    b = b_ref[:, :]
    o_ref[:, :] = jnp.matmul(a, b)


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        block_m = 128
        block_n = 128

        grid_m = M // block_m
        grid_n = N // block_n

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(A, B)

    def set_weights(self, weights_dict):
        pass


M = 256
N = 256
K = 131072 * 4


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]


def get_init_inputs():
    return []
