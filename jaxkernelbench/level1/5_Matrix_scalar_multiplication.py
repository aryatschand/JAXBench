"""
JAXBench Level 1 - Task 5: Matrix_scalar_multiplication
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.122693
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        pass
    
    def forward(self, A: jnp.ndarray, s: float) -> jnp.ndarray:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return A * s

M = 16384 * 4
N = 4096 * 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    A = jax.random.uniform(key, shape=(M, N))
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed