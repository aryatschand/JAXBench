import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    acc = jnp.dot(x, y, preferred_element_type=jnp.float32)
    o_ref[...] = acc.astype(o_ref.dtype)

class Model:
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (jnp.ndarray): Input 4D tensor of shape (b, i, j, l)
        B (jnp.ndarray): Input matrix of shape (l, k)

    Returns:
        jnp.ndarray: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        pass

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (jnp.ndarray): Input 4D tensor of shape (b, i, j, l)
            B (jnp.ndarray): Input matrix of shape (l, k)

        Returns:
            jnp.ndarray: Output 4D tensor of shape (b, i, j, k)
        """
        M = 1
        for dim in A.shape[:-1]:
            M *= dim
        K = A.shape[-1]
        N = B.shape[-1]
        
        A_2d = A.reshape((M, K))
        
        BM = min(M, 1024)
        while M % BM != 0:
            BM //= 2
            
        BN = min(N, 256)
        while N % BN != 0:
            BN //= 2
            
        grid_shape = (M // BM, N // BN)
        
        out_shape = jax.ShapeDtypeStruct((M, N), A.dtype)
        
        C_2d = pl.pallas_call(
            matmul_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((
