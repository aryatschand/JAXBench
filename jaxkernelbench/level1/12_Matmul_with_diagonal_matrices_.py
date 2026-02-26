"""
JAXBench Level 1 - Task 12: Matmul_with_diagonal_matrices_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.125342
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): A 1D array representing the diagonal of the diagonal matrix. Shape: (N,).
            B (jnp.ndarray): A 2D array representing the second matrix. Shape: (N, M).

        Returns:
            jnp.ndarray: The result of the matrix multiplication. Shape: (N, M).
        """
        # Logically equivalent to jnp.diag(A) @ B
        # more efficient as no need to materialize a full N×N matrix
        return jnp.expand_dims(A, 1) * B

M = 4096
N = 4096

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N,))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed