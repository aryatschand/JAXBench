import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_transpose_a_kernel(a_ref, b_ref, o_ref):
    a = a_ref[...]      # (K, block_m)
    b = b_ref[...]      # (K, block_n)
    o_ref[...] = jnp.matmul(a.T, b)  # (block_m, block_n)


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication.

        Args:
            A: Input array of shape (M, K).
            B: Input array of shape (K, N).

        Returns:
            Output array of shape (M, N).
        """
        K, M = A.shape
        Kb, N = B.shape
        assert K == Kb

        block_m = 128
        block_n = 128

        grid_m = M // block_m
        grid_n = N // block_n

        return pl.pallas_call(
            matmul_transpose_a_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((K, block_m), lambda i, j: (0, i)),
                    pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
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
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]


def get_init_inputs():
    return []
