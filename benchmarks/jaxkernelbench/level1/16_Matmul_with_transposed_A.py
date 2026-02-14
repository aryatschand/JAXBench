"""
JAXBench Level 1 - Task 16: Matmul_with_transposed_A
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.126717
"""

import jax
import jax.numpy as jnp

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
        return jnp.matmul(A.T, B)

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(K, M))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed