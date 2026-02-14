"""
JAXBench Level 1 - Task 98: KLDivLoss
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.150972
"""

import jax
import jax.numpy as jnp
from jax.nn import softmax

class Model:
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        # KL divergence implementation
        log_predictions = jnp.log(predictions)
        return jnp.mean(targets * (jnp.log(targets) - log_predictions))

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    scale = jax.random.uniform(key1)
    pred = softmax(jax.random.uniform(key2, (batch_size, *input_shape)) * scale, axis=-1)
    tgt = softmax(jax.random.uniform(key3, (batch_size, *input_shape)), axis=-1)
    return [pred, tgt]

def get_init_inputs():
    return []