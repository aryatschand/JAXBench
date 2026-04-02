```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bmm_kernel(a_ref, b_ref, c_ref):
    # a_ref shape: (1, block_m, k)
    # b_ref shape: (1, k, block_n)
    # c_ref shape: (1, block_m, block_n)
    
    # Perform matrix multiplication for the block.
    # XLA will automatically tile this dot product over the MXU and manage the accumulator.
    c_ref[0, :, :] = jnp.dot(
        a_ref[0, :, :], 
        b_ref[0, :, :], 
        preferred_element_type=jnp.float32
    ).astype(c_ref.dtype)

class Model:
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        pass
    
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs batched matrix multiplication using a Pallas TPU kernel.

        Args:
            A: Input array of shape (batch_size, m, k).
            B: Input array of shape (batch_size, k, n).

        Returns:
            C: Output array of shape (batch_size, m, n).
        """
        batch_size, m, k = A.shape
        _, _, n = B.shape
        
        # Tile over M and N dimensions. K is fully loaded into VMEM since it fits (1024 elements).
        block_m = 256
        block_n = 256
        
        # Ensure block sizes evenly divide the dimensions
        block_m = min(block_m, m)
        block_n = min(block_n, n)
        
        grid_shape = (batch_size, m // block_m, n // block_n)
        
        return pl.pallas_call(
            bmm_kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size, m, n), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((1, block_m, k), lambda b, i, j: (b, i, 0)),
                    pl.BlockSpec((1, k, block_n), lambda b, i, j: (b, 0, j)),
                ],
                out_specs=
