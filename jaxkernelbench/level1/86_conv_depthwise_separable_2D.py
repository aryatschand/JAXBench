"""
JAXBench Level 1 - Task 86: conv_depthwise_separable_2D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.146954
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # Initialize shapes for depthwise conv
        # For depthwise conv with groups=in_channels, each filter operates on 1 channel
        # Shape: (kH, kW, 1, in_channels) for HWIO format with feature_group_count
        depthwise_shape = (kernel_size, kernel_size, 1, in_channels)
        self.depthwise_weight = jnp.zeros(depthwise_shape)
        self.depthwise_bias = jnp.zeros((in_channels,)) if bias else None
        
        # Initialize shapes for pointwise conv
        pointwise_shape = (1, 1, in_channels, out_channels)
        self.pointwise_weight = jnp.zeros(pointwise_shape)
        self.pointwise_bias = jnp.zeros((out_channels,)) if bias else None
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.bias = bias
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'depthwise.weight' in name:
                # PyTorch depthwise weight shape: (in_channels, 1, kH, kW) with groups=in_channels
                # Need to convert to JAX format: (kH, kW, 1, in_channels) for HWIO
                # value shape is (in_channels, 1, kH, kW)
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                self.depthwise_weight = value
            elif 'pointwise.weight' in name:
                # Convert from (out_channels, in_channels, 1, 1) to (1, 1, in_channels, out_channels)
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                self.pointwise_weight = value
            elif 'depthwise.bias' in name:
                self.depthwise_bias = jnp.array(value)
            elif 'pointwise.bias' in name:
                self.pointwise_bias = jnp.array(value)

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Depthwise convolution
        # With feature_group_count=in_channels, the kernel shape should be (kH, kW, 1, in_channels)
        # where the input feature dim (1) * feature_group_count = in_channels
        x = lax.conv_general_dilated(
            x,
            self.depthwise_weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.in_channels,
            rhs_dilation=(self.dilation, self.dilation)
        )
        if self.depthwise_bias is not None:
            x = x + self.depthwise_bias.reshape(1, 1, 1, -1)

        # Pointwise convolution
        x = lax.conv_general_dilated(
            x,
            self.pointwise_weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        if self.pointwise_bias is not None:
            x = x + self.pointwise_bias.reshape(1, 1, 1, -1)
            
        # Convert back from NHWC to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x

# Test parameters
batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]