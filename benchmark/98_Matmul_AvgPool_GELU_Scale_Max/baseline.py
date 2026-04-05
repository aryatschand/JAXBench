"""
JAXBench Level 2 - Matmul_AvgPool_GELU_Scale_Max
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu

class Model:
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.weight) + self.bias
        
        # Reshape for 1D pooling
        x = jnp.expand_dims(x, axis=1)
        
        # AvgPool1d using reduce_window
        x = jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, 1, self.pool_kernel_size),
            window_strides=(1, 1, self.pool_kernel_size),
            padding='VALID'
        ) / self.pool_kernel_size
        
        # Remove pooling dimension
        x = jnp.squeeze(x, axis=1)
        
        # GELU activation
        x = gelu(x)
        
        # Scale
        x = x * self.scale_factor
        
        # Max along dim 1
        x = jnp.max(x, axis=1)
        
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]