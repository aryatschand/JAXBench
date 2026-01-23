"""
JAXBench Level 1 - Task 10: 3D_tensor_matrix_multiplication
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.124561
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (jnp.ndarray): Input 3D tensor of shape (N, M, K).
            B (jnp.ndarray): Input matrix of shape (K, L).

        Returns:
            jnp.ndarray: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return jnp.matmul(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, M, K))
    B = jax.random.uniform(key2, shape=(K, L))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed