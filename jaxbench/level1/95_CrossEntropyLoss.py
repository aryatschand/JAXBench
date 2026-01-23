"""
JAXBench Level 1 - Task 95: CrossEntropyLoss
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.150314
"""

import jax
import jax.numpy as jnp
from jax.nn import log_softmax

class Model:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        # Compute log probabilities
        log_probs = log_softmax(predictions)
        # Get probs for the target classes
        target_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=1)
        # Average negative log likelihood
        return -jnp.mean(target_probs)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    return [
        jax.random.uniform(key1, shape=(batch_size, *input_shape)),
        jax.random.randint(key2, shape=(batch_size,), minval=0, maxval=num_classes)
    ]

def get_init_inputs():
    return []