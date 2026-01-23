"""
JAXBench Level 1 - Task 96: HuberLoss (reduced size to avoid memory issues)
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp


class Model:
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        pass
    
    def set_weights(self, weights_dict):
        """No weights for loss function."""
        pass
    
    def forward(self, predictions, targets):
        """
        Compute Smooth L1 (Huber) loss.
        
        For |x| < 1: 0.5 * x^2
        For |x| >= 1: |x| - 0.5
        """
        diff = predictions - targets
        abs_diff = jnp.abs(diff)
        
        # Smooth L1 loss formula
        loss = jnp.where(
            abs_diff < 1.0,
            0.5 * diff ** 2,
            abs_diff - 0.5
        )
        
        return jnp.mean(loss)


# Test code - REDUCED SIZE to avoid memory issues
batch_size = 4096    # Reduced from 32768
input_shape = (4096,)  # Reduced from (32768,)
dim = 1


def get_inputs():
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    scale = jax.random.uniform(key1, ())
    predictions = jax.random.uniform(key2, (batch_size, *input_shape)) * scale
    targets = jax.random.uniform(key3, (batch_size, *input_shape))
    return [predictions, targets]


def get_init_inputs():
    return []

