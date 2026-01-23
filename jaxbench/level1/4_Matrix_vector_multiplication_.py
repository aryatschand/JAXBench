"""
JAXBench Level 1 - Task 4: Matrix_vector_multiplication_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.122005
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return jnp.matmul(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, 1))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed