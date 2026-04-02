"""
JAXBench Level 1 - Task 17: Matmul_with_transposed_B
Pallas TPU kernel version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    a_block = a_ref[:, :]          # (bm, K)
    b_block = b_ref[:, :]          # (bn, K)
    o_ref[:, :] = jnp.matmul(a_block, jnp.swapaxes(b_block, 0, 1))


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        M, K = A.shape
        N, _ = B.shape

        bm = 128
        bn = 128

        grid = (M // bm, N // bn)

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((bn, K), lambda i, j: (j, 0)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(N, K))
    return [A, B]


def get_init_inputs():
    return []
