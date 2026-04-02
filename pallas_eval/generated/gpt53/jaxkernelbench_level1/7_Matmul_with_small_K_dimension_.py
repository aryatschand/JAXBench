import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    a = a_ref[:, :]          # (BM, K)
    b = b_ref[:, :]          # (K, BN)
    acc = jnp.dot(a.astype(jnp.float32), b.astype(jnp.float32))
    o_ref[:, :] = acc.astype(o_ref.dtype)


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        BM = 128
        BN = 128

        grid = (M // BM, N // BN)

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(A, B)


M = 16384 * 2
N = 16384 * 2
K = 32 * 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]


def get_init_inputs():
    return []
