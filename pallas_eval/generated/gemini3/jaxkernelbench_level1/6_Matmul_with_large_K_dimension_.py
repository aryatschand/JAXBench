```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref):
    # Load blocks from VMEM
    a = a_ref[...]
    b = b_ref[...]
    
    # Compute matrix multiplication for the tile.
    # The Mosaic compiler will automatically tile this dot product over the K dimension
    # (which is bk=4096) and keep the accumulator in scratch VMEM.
    c = jnp.dot(a, b, preferred_element_type=jnp.float32)
    
    # Write the result to the output reference
    c_ref[0, ...] = c

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of A and B using a Pallas TPU kernel.

        Args:
            A: Input array of shape (M, K)
            B: Input array of shape (K, N)

        Returns:
            Output array of shape (M, N)
        """
        M, K = A.shape
        _, N = B.shape
        
        # Tile sizes
        bm = min(M, 256)
        bn = min(N, 256)
        bk = min(K, 4096)
        
        # Grid dimensions
        grid_m = M // bm
        grid_n = N // bn
        grid_k = K // bk
        
        grid = (grid_m, grid_n, grid_k)
        
        # We compute partial sums over the K dimension and store them in a 3D array.
        # This avoids HBM round-trips for the accumulator during the K-loop.
        out_shape = jax.ShapeDtypeStruct((grid_k, M, N), jnp.float32)
        
        C_partial = pl.pallas_call(
            matmul_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                ],
                out_specs=pl.BlockSpec((1, bm, bn), lambda i, j, k: (k, i, j)),
            ),
        )(A, B)
