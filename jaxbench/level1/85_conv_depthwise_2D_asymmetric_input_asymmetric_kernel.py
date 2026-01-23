"""
JAXBench Level 1 - Task 85: conv_depthwise_2D_asymmetric_input_asymmetric_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.146452
"""

import jax
import jax.numpy as jnp
import jax.lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = ((padding_h, padding_h), (padding_w, padding_w))
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights with correct shape and transpose for JAX
        weight_shape = (in_channels, 1, kernel_size_h, kernel_size_w)
        k = jax.random.PRNGKey(0)
        weight = jax.random.normal(k, weight_shape) * 0.02
        self.weight = jnp.transpose(weight, (2, 3, 1, 0))  # HWIO format
        
        if bias:
            self.bias = jnp.zeros(in_channels)
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Convert from PyTorch OIHW to JAX HWIO format
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                setattr(self, 'weight', value)
            elif 'bias' in name:
                setattr(self, 'bias', jnp.array(value))
            else:
                setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Depthwise conv using JAX
        y = jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.in_channels
        )
        
        if self.use_bias:
            y = y + self.bias
            
        # Convert back to NCHW
        return jnp.transpose(y, (0, 3, 1, 2))

# Test parameters
batch_size = 32
in_channels = 128  
out_channels = 128
kernel_size_h = 3
kernel_size_w = 7
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]