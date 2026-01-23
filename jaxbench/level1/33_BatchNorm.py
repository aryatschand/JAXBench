"""
JAXBench Level 1 - Task 33: BatchNorm
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.132818
"""

import jax
import jax.numpy as jnp
from jax.nn import initializers

class Model:
    def __init__(self, num_features: int):
        self.num_features = num_features
        # Initialize with dummy values - will be overwritten by set_weights()
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.running_mean = jnp.zeros(num_features)
        self.running_var = jnp.ones(num_features)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Apply batch norm using running statistics (inference mode)
        x_normalized = (x - self.running_mean.reshape(1, 1, 1, -1)) / jnp.sqrt(self.running_var.reshape(1, 1, 1, -1) + self.eps)
        out = self.weight.reshape(1, 1, 1, -1) * x_normalized + self.bias.reshape(1, 1, 1, -1)
        
        # Convert back from NHWC to NCHW
        return jnp.transpose(out, (0, 3, 1, 2))

batch_size = 64
features = 64
dim1 = 512 
dim2 = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]