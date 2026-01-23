"""
JAXBench Level 1 - Task 9: Tall_skinny_matrix_multiplication_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.124242
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (jnp.ndarray): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            jnp.ndarray: Output matrix of shape (M, N) or (N, M)
        """
        return jnp.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, N))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed