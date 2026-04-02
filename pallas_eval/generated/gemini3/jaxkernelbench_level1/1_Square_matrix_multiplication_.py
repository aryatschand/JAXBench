import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref):
    # Read entire blocks from HBM to VMEM
    a = a_ref[...]
    b = b_ref[...]
    
    # Perform matrix multiplication with f32 accumulation
    c = jnp.matmul(a, b, preferred_element_type=jnp.float32)
    
    # Write the result back to HBM
    c_ref[...] = c.astype(c_ref.dtype)

class Model:
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A: Input matrix A of shape (N, N).
            B: Input matrix B of shape (N, N).

        Returns:
            Output matrix C of shape (N, N).
        """
        BM, BN = 256, 256
        M, K = A.shape
        _, N = B.shape
        
        # Fallback to standard matmul if dimensions are not perfectly divisible by block sizes
        if M % BM != 0 or N % BN != 0:
            return jnp.matmul(A, B)
            
        grid_shape = (M // BM, N // BN)
        
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(A, B)

N = 2048 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, N))
    B = jax.random.uniform(key2, shape=(N, N))
    return [A, B]

def get_init_inputs():
    return []
