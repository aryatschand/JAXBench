import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def diag_matmul_kernel(a_ref, b_ref, c_ref):
    a_val = a_ref[...]
    b_val = b_ref[...]
    a_val_rep = pltpu.repeat(a_val, b_val.shape[1], axis=1)
    c_ref[...] = a_val_rep * b_val

class Model:
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): A 1D array representing the diagonal of the diagonal matrix. Shape: (N,).
            B (jnp.ndarray): A 2D array representing the second matrix. Shape: (N, M).

        Returns:
            jnp.ndarray: The result of the matrix multiplication. Shape: (N, M).
        """
        N_dim = A.shape[0]
        M_dim = B.shape[1]
        
        A_2d = jnp.expand_dims(A, 1)
        
        block_N = min(N_dim, 512)
        block_M = min(M_dim, 512)
        
        grid_shape = (N_dim // block_N, M_dim // block_M)
        
        return pl.pallas_call(
            diag_matmul_kernel,
            out_shape=jax.ShapeDtypeStruct(B.shape, B.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_N, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_N, block_M), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_N, block_M), lambda i, j: (i, j)),
            ),
        )(A_2d, B)

M = 4096
N = 4096

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N,))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []
