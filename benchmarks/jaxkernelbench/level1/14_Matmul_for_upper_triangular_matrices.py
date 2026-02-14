"""
JAXBench Level 1 - Task 14: Matmul_for_upper_triangular_matrices
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.126068
"""

import jax
import jax.numpy as jnp

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
        return jnp.triu(jnp.matmul(A, B))

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jnp.triu(jax.random.uniform(key1, (N, N)))
    B = jnp.triu(jax.random.uniform(key2, (N, N)))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []