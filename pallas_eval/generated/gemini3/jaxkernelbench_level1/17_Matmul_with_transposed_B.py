import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 128
BN = 128
BK = 256

def matmul_kernel(a_ref, b_ref, c_ref):
    a_val = a_ref[...]
    b_val = b_ref[...]
    
    K_dim = a_val.shape[1]
    
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    def body(k, acc):
        a_block = jax.lax.dynamic_slice(a_val, (0, k * BK), (BM, BK))
        b_block = jax.lax.dynamic_slice(b_val, (0, k * BK), (BN, BK))
        return acc + jax.lax.dot_general(
            a_block, b_block,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32
        )
    
    acc = jax.lax.fori_loop(0, K_dim // BK, body, acc)
    c_ref[...] = acc.astype(c_ref.dtype)

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
            B: Input array of shape (N, K).

        Returns:
            Output array of shape (M, N).
        """
        M, K = A.shape
        N, _ = B.shape
        
        grid_shape = (M // BM, N // BN)
        
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((BN, K), lambda i, j: (j, 0)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
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
