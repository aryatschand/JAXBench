"""
JAXBench Level 1 - Task 1: Square_matrix_multiplication_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.119679
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A: Input matrix A of shape (N, N).
            B: Input matrix B of shape (N, N).

        Returns:
            Output matrix C of shape (N, N).
        """
        return jnp.matmul(A, B)

N = 2048 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, N))
    B = jax.random.uniform(key2, shape=(N, N))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed