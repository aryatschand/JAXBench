"""
JAXBench Level 1 - Task 100: HingeLoss
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.151628
"""

import jax
import jax.numpy as jnp

class Model:
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        return jnp.mean(jnp.maximum(1 - predictions * targets, 0))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    return [
        jax.random.uniform(key1, shape=(batch_size, *input_shape)),
        jax.random.randint(key2, shape=(batch_size,), minval=0, maxval=2) * 2 - 1
    ]

def get_init_inputs():
    return []