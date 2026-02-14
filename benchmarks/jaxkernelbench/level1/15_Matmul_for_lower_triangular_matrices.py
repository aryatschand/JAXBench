"""
JAXBench Level 1 - Task 15: Matmul_for_lower_triangular_matrices
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.126395
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (jnp.ndarray): Lower triangular matrix of shape (N, N).
            B (jnp.ndarray): Lower triangular matrix of shape (N, N).

        Returns:
            jnp.ndarray: The result of matrix multiplication C of shape (N, N).
        """
        return jnp.tril(jnp.matmul(A, B))

M = 4096

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, (M, M))
    B = jax.random.uniform(key2, (M, M))
    A = jnp.tril(A)
    B = jnp.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed