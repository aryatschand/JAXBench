"""
JAXBench Level 1 - Task 15: Matmul_for_lower_triangular_matrices
Pallas TPU kernel version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_tril_kernel(A_ref, B_ref, O_ref):
    A = A_ref[:, :]
    B = B_ref[:, :]
    C = jnp.matmul(A, B)
    O_ref[:, :] = jnp.tril(C)


class Model:
    """
    Simple model that performs a matrix multiplication (C = A * B)
    where A and B are lower triangular matrices.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        N = A.shape[0]

        return pl.pallas_call(
            matmul_tril_kernel,
            out_shape=jax.ShapeDtypeStruct((N, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1, 1),
                in_specs=[
                    pl.BlockSpec((N, N), lambda i, j: (0, 0)),
                    pl.BlockSpec((N, N), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((N, N), lambda i, j: (0, 0)),
            ),
        )(A, B)


M = 4096


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, (M, M))
    B = jax.random.uniform(key2, (M, M))
    A = jnp.tril(A)
    B = jnp.tril(B)
    return [A, B]


def get_init_inputs():
    return []
