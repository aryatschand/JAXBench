import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref):
    a = a_ref[...]
    b = b_ref[...]
    c_ref[...] = jnp.dot(a, b, preferred_element_type=jnp.float32)

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        M, K = A.shape
        _, N = B.shape
        
        def get_block_size(dim, max_size=512):
            for size in [max_size, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
                if dim % size == 0:
                    return size
            return 1
            
        BM = get_block_size(M, 512)
        BN = get_block_size(N, 512)
        BK = K
        
        grid_shape = (M // BM, N // BN)
        
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, BK), lambda i, j: (i, 0)),
                    pl.BlockSpec((BK, BN), lambda i, j: (0, j)),
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
