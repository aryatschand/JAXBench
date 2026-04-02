import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        N, M, K = A.shape
        K2, L = B.shape
        assert K == K2

        A2 = A.reshape(N * M, K)

        block_m = 128
        block_n = 128

        def kernel(a_ref, b_ref, o_ref):
            a = a_ref[:, :]
            b = b_ref[:, :]
            o_ref[...] = jnp.matmul(a, b)

        grid_m = (N * M) // block_m
        grid_n = L // block_n

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N * M, L), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(A2, B)

        return out.reshape(N, M, L)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, M, K))
    B = jax.random.uniform(key2, shape=(K, L))
    return [A, B]

def get_init_inputs():
    return []
