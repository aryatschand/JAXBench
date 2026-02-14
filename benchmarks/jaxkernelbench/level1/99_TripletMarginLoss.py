"""
JAXBench Level 1 - Task 99: TripletMarginLoss
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.151254
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, margin=1.0):
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # PyTorch TripletMarginLoss uses L2 distance (Euclidean norm), not squared distance
        # Default p=2, so it computes ||a - p||_2 and ||a - n||_2
        d_pos = jnp.sqrt(jnp.sum((anchor - positive) ** 2, axis=-1) + 1e-12)
        d_neg = jnp.sqrt(jnp.sum((anchor - negative) ** 2, axis=-1) + 1e-12)
        
        # Compute triplet loss with margin
        loss = jnp.maximum(0.0, d_pos - d_neg + self.margin)
        return jnp.mean(loss)

batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    scale = jax.random.uniform(key1)
    anchor = jax.random.uniform(key2, (batch_size,) + input_shape) * scale
    positive = jax.random.uniform(key3, (batch_size,) + input_shape)
    negative = jax.random.uniform(key4, (batch_size,) + input_shape)
    
    return [anchor, positive, negative]

def get_init_inputs():
    return [1.0]  # Default margin