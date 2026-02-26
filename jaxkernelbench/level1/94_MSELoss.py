"""
JAXBench Level 1 - Task 94: MSELoss
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.149871
"""

import jax
import jax.numpy as jnp

class Model:
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    scale = jax.random.uniform(key1, shape=())
    return [jax.random.uniform(key2, shape=(batch_size, *input_shape))*scale,
            jax.random.uniform(key3, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return []