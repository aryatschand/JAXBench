import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    optimized for tall-skinny shapes using a TPU Pallas kernel.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        bm = 128
        bn = 128

        def kernel(a_ref, b_ref, o_ref):
            a = a_ref[...]          # (bm, K)
            b = b_ref[...]          # (K, bn)
            o_ref[...] = jnp.matmul(a, b)

        grid_m = M // bm
        grid_n = N // bn

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, N))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []
