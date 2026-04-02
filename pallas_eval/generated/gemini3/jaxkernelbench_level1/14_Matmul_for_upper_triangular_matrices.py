```python
"""
JAXBench Level 1 - Task 14: Matmul_for_upper_triangular_matrices
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.126068
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 256
BN = 256
BK = 256

def matmul_triu_kernel(a_ref, b_ref, c_ref):
    i = pl.program_id(0)
    j = pl.program_id(1)
    
    # Initialize accumulator in scratch VMEM
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    def loop_body(k, acc_inner):
        # Slice the required tiles from the row/col panels loaded in VMEM
        a_tile = a_ref[:, k * BK : k * BK + BK]
        b_tile = b_ref[k * BK : k * BK + BK, :]
        return acc_inner + jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)
    
    # For upper triangular matrices, we only need to compute k from i to j.
    # If i > j, lower >= upper, the loop executes 0 times and returns zeros.
    acc = jax.lax.fori_loop(i, j + 1, loop_body, acc)
    
    # Mask out elements strictly below the diagonal for blocks on the diagonal
    row_idx = i * BM + jnp.arange(BM)[:, None]
    col_idx = j * BN + jnp.arange(BN)[None, :]
    mask = row_idx <= col_idx
    acc = jnp.where(mask, acc, 0.0)
    
    c_ref[:, :] = acc.astype(c_ref.dtype)

class Model:
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (jnp.ndarray): Upper triangular matrix of shape (N, N).
            B (jnp.ndarray): Upper triangular matrix of shape (N, N).

        Returns:
            jnp.ndarray: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        n = A.shape[0]
        grid_shape = (n // BM,
