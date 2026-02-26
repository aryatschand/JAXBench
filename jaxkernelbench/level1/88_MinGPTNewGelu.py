"""
JAXBench Level 1 - Task 88: MinGPTNewGelu
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.147596
"""

import jax
import jax.numpy as jnp
import math

class Model:
    def __init__(self):
        pass
    
    def forward(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass

batch_size = 8192
dim = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, dim))]

def get_init_inputs():
    return []