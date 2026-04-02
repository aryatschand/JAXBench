import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 256
BN = 256

def matmul_kernel(a_ref, b_ref, c_ref):
    c_ref[...] = jnp.matmul(
        a_ref[...], 
        b_ref[...], 
        preferred_element_type=jnp.float32
    ).astype(c_ref.dtype)

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        M, K = A.shape
        _, N = B.shape
        
        # Pad dimensions to multiples of block sizes
        M_pad = (M + BM - 1) // BM * BM
        N_pad = (N + BN - 1) // BN * BN
        # Pad K to a multiple of 128 for optimal MXU utilization
        K_pad = (K + 127) // 128 * 128
        
        A_pad = jnp.pad(A, ((0, M_pad - M), (0, K_pad - K)))
        B_pad = jnp.pad(B, ((0, K_pad - K), (0, N_pad - N)))
        
        grid_shape = (M_pad // BM, N_pad // BN)
        
        C_pad = pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M_pad, N_pad), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, K_pad), lambda i, j: (i, 0)),
                    pl.BlockSpec((K_pad, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(A_pad, B_pad)
        
        # Slice back to original irregular shape
        return C_pad[:M, :N]

M = 8205
K = 2949
N = 5921

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]

def get_init_inputs():
    return []
