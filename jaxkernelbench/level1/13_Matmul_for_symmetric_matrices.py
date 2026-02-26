"""
JAXBench Level 1 - Task 13: Matmul_for_symmetric_matrices
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.125745
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (jnp.ndarray): Input matrix A, shape (N, N), symmetric.
            B (jnp.ndarray): Input matrix B, shape (N, N), symmetric.

        Returns:
            jnp.ndarray: Output matrix C, shape (N, N).
        """
        return jnp.matmul(A, B)

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric arrays A and B.
    """
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    
    A = jax.random.uniform(key1, shape=(N, N))
    A = (A + A.T) / 2  # Ensure symmetry
    
    B = jax.random.uniform(key2, shape=(N, N))
    B = (B + B.T) / 2  # Ensure symmetry
    
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []