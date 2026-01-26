"""
JAXBench Level 2 - Matmul_MaxPool_Sum_Scale
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        # PyTorch Linear weight is (out_features, in_features), need to transpose for matmul
        self.matmul_weight = jnp.zeros((out_features, in_features))
        self.matmul_bias = jnp.zeros(out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer: PyTorch stores weight as (out_features, in_features)
        # So we need x @ weight.T + bias
        x = x @ self.matmul_weight.T + self.matmul_bias
        
        # MaxPool1d - PyTorch expects (N, C, L) format
        # After unsqueeze(1), shape is (batch_size, 1, out_features)
        x = jnp.expand_dims(x, axis=1)  # Add channel dim: (N, 1, L)
        
        # For reduce_window, we need window on the last dimension
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 1, self.kernel_size),
            window_strides=(1, 1, self.kernel_size),
            padding='VALID'
        )
        
        x = jnp.squeeze(x, axis=1)  # Remove channel dim: (N, L//kernel_size)
        
        # Sum and scale
        x = jnp.sum(x, axis=1)
        x = x * self.scale_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]