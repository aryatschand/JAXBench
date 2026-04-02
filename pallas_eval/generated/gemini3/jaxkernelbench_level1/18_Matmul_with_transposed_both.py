```python
"""
JAXBench Level 1 - Task 18: Matmul_with_transposed_both
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.127350
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
        M = A.shape[1]
        K = A.shape[0]
        N = B.shape[0]
        
        # Block sizes chosen to fit comfortably in TPU VMEM
        BM = 128
        BN = 128
        BK = 256
        
        grid_shape = (M // BM, N // BN)
        
        def matmul_kernel(A_ref, B_ref, C_ref):
            # A_ref has shape (K, BM)
            # B_ref has shape (BN, K)
            # C_ref has shape (BM, BN)
            
            acc = jnp.zeros((BM, BN), dtype=jnp.float32)
            
            def body(i, acc):
                # Slice chunks of size BK along the K dimension
                a = A_ref[i * BK : (i + 1) * BK, :]  # shape: (BK, BM)
                b = B_ref[:, i * BK : (i + 1) * BK]  # shape: (BN, BK)
                
                # a.T is (BM, BK), b.T is (BK, BN)
                # Result is (BM, BN)
                return acc + jnp.matmul(a.T, b.T, preferred_element_type=jnp.float32)
            
            acc = jax.lax.fori_loop(0, K // BK, body, acc)
            C_ref[:, :] = acc.astype(C_ref.dtype)

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in
