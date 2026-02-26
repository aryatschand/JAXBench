"""
JAXBench Level 1 - Task 3: Batched_matrix_multiplication
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.121700
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        pass
    
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input array of shape (batch_size, m, k).
            B: Input array of shape (batch_size, k, n).

        Returns:
            C: Output array of shape (batch_size, m, n).
        """
        return jnp.einsum('bij,bjk->bik', A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(batch_size, m, k))
    B = jax.random.uniform(key2, shape=(batch_size, k, n))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed