"""
JAXBench Level 1 - Task 2: Standard_matrix_multiplication_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.121069
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(x_ref, y_ref, z_ref):
    # The Mosaic compiler automatically tiles this dot product over the K dimension
    # and pipelines the DMA loads from HBM to VMEM.
    z_ref[...] = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication using a Pallas TPU kernel.

        Args:
            A: Input array of shape (M, K).
            B: Input array of shape (K, N).

        Returns:
            Output array of shape (M, N).
        """
        # 128x128 is the native MXU size on TPU.
        # The blocks (128, K) and (K, 128) will fit comfortably in the 16MB VMEM.
        bm, bn = 128, 128
        M, K = A.shape
        _, N = B.shape
        
        grid_shape = (M // bm, N // bn)
        
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]

def get_init_inputs():
    return []
