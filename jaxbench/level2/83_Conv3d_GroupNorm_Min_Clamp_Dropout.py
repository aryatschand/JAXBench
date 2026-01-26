"""
JAXBench Level 2 - Conv3d_GroupNorm_Min_Clamp_Dropout
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import jax.random as random

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        # Conv3d weights
        self.conv_weight = jnp.zeros((out_channels, in_channels, *self.kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        
        # GroupNorm parameters
        self.norm_weight = jnp.ones(out_channels)
        self.norm_bias = jnp.zeros(out_channels)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Conv3d - no padding (PyTorch default)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))  # (out,in,D,H,W) -> (D,H,W,in,out)
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',  # PyTorch Conv3d default has no padding
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
        x = x + self.conv_bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # GroupNorm
        N, C, D, H, W = x.shape
        G = self.groups
        x = x.reshape(N, G, C//G, D, H, W)
        mean = jnp.mean(x, axis=(2,3,4,5), keepdims=True)
        var = jnp.var(x, axis=(2,3,4,5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, D, H, W)
        x = x * self.norm_weight.reshape(1,-1,1,1,1) + self.norm_bias.reshape(1,-1,1,1,1)

        # Min and clamp
        x = jnp.minimum(x, self.min_value)
        x = jnp.clip(x, self.min_value, self.max_value)
        
        # Dropout - in inference mode (no dropout applied)
        # For training, you would need to pass an rng key
        # Since the test framework doesn't pass rng, we skip dropout (inference mode)
        # x = x  # No-op for inference
        
        return x

batch_size = 128
in_channels = 3  
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]