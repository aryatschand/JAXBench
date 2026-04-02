import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    a_block = a_ref[...]            # (K, block_m)
    b_block = b_ref[...]            # (block_n, K)
    a_t = jnp.transpose(a_block)    # (block_m, K)
    b_t = jnp.transpose(b_block)    # (K, block_n)
    o_ref[...] = jnp.matmul(a_t, b_t)


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass

    def forward(self, A, B):
        K, M = A.shape
        N, _ = B.shape

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
                    pl.BlockSpec((K, block_m), lambda i, j: (0, i)),
                    pl.BlockSpec((block_n, K), lambda i, j: (j, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(A, B)

    def set_weights(self, weights_dict):
        pass


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(K, M))
    B = jax.random.uniform(key2, shape=(N, K))
    return [A, B]


def get_init_inputs():
    return []
