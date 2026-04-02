```python
"""
JAXBench Level 2 - Conv3d_GroupNorm_Mean
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_gn_mean_kernel(L, pad_L, C_per_G):
    def gn_mean_kernel(x_ref, gamma_ref, beta_ref, out_ref):
        x = x_ref[0, 0, :, :]
        gamma = gamma_ref[0, :].reshape(C_per_G, 1)
        beta = beta_ref[0, :].reshape(C_per_G, 1)
        
        valid_mask = jnp.arange(pad_L) < L
        valid_mask = valid_mask.reshape(1, pad_L)
        
        x_valid = jnp.where(valid_mask, x, 0.0)
        num_elements = C_per_G * L
        mean = jnp.sum(x_valid) / num_elements
        
        diff = jnp.where(valid_mask, x - mean, 0.0)
        var = jnp.sum(diff * diff) / num_elements
        
        x_norm = diff / jnp.sqrt(var + 1e-5)
        x_scaled = x_norm * gamma + beta
        
        group_sum = jnp.sum(jnp.where(valid_mask, x_scaled, 0.0))
        out_ref[0, 0] = group_sum
    return gn_mean_kernel

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        # Conv3d weights shape: (out_channels, in_channels, D, H, W)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        
        # GroupNorm parameters
        self.num_groups = num_groups
        self.gamma = jnp.ones(out_channels)
        self.beta = jnp.zeros(out_channels)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Conv3d optimized layout
        x = jax.lax.conv_general_dilated(
            x, self.weight,
            window_strides=(1, 1
