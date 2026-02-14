"""
JAXBench Level 1 - Task 8: Matmul_with_irregular_shapes_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.123757
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input array with shape (M, K).
            B: Input array with shape (K, N).

        Returns:
            C: Output array with shape (M, N).
        """
        return jnp.matmul(A, B)

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
    return []  # No special initialization inputs needed