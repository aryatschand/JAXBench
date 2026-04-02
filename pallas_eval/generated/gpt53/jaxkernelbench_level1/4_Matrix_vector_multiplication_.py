import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        pass

    def forward(self, A, B):
        M, K = A.shape

        block_m = 128
        block_k = 1024

        assert M % block_m == 0
        assert K % block_k == 0

        grid_m = M // block_m
        grid_k = K // block_k

        def kernel(a_ref, b_ref, o_ref):
            a = a_ref[:, :]
            b = b_ref[:, :]
            partial = jnp.matmul(a, b)
            o_ref[:, :] += partial

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, 1), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_k),
                in_specs=[
                    pl.BlockSpec((block_m, block_k), lambda i, j: (i, j)),
                    pl.BlockSpec((block_k, 1), lambda i, j: (j, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
            ),
        )(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, 1))
    return [A, B]

def get_init_inputs():
    return []
