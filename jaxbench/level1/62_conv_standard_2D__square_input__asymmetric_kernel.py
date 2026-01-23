"""
JAXBench Level 1 - Task 62: conv_standard_2D__square_input__asymmetric_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.141930
"""

import jax
import jax.numpy as jnp
import jax.lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        rng = jax.random.PRNGKey(0)
        k_h, k_w = kernel_size
        
        # Initialize weights with same shape as PyTorch but transpose for JAX
        weight_shape = (out_channels, in_channels, k_h, k_w)
        weight = jax.random.normal(rng, weight_shape) * 0.02
        self.weight = jnp.transpose(weight, (2, 3, 1, 0))  # HWIO format
        
        self.bias = None
        if bias:
            self.bias = jnp.zeros(out_channels)
            
        if isinstance(padding, int):
            self.padding = [(padding, padding), (padding, padding)]
        else:
            self.padding = [(p, p) for p in padding]
            
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Convert from PyTorch OIHW to JAX HWIO
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                setattr(self, 'weight', value)
            elif 'bias' in name:
                setattr(self, 'bias', jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        out = jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups
        )
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, -1)
            
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

# Test code
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512

def get_inputs():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]