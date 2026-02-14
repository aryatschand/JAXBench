"""
JAXBench Level 1 - Task 73: conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.143749
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights with correct shape for ConvTranspose3d
        # PyTorch shape: (in_channels, out_channels // groups, kD, kH, kW)
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.conv_transpose3d_weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.conv_transpose3d_bias = jnp.zeros(out_channels)
        else:
            self.conv_transpose3d_bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose weight from (in, out, D, H, W) to (D, H, W, out, in)
        weight = jnp.transpose(self.conv_transpose3d_weight, (2, 3, 4, 1, 0))
        
        # For conv_transpose in JAX, we need to use padding that matches PyTorch's behavior
        # PyTorch ConvTranspose3d output size: (input - 1) * stride - 2 * padding + kernel_size + output_padding
        # JAX conv_transpose with padding='SAME' or explicit padding
        
        # Use low-level padding specification to match PyTorch
        # For transposed convolution, the effective padding is different
        # We need to pad the output, not the input
        padding_lax = ((self.padding, self.padding),
                       (self.padding, self.padding),
                       (self.padding, self.padding))
        
        if self.groups == 1:
            out = lax.conv_transpose(
                x,
                weight,
                strides=(self.stride, self.stride, self.stride),
                padding=padding_lax,
                dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
                transpose_kernel=True
            )
        else:
            # Handle grouped convolution manually
            batch_size = x.shape[0]
            in_channels_per_group = self.in_channels // self.groups
            out_channels_per_group = self.out_channels // self.groups
            
            # Split input along channel dimension
            x_groups = jnp.split(x, self.groups, axis=-1)
            
            # Split weight along input channel dimension (last dim after transpose)
            weight_groups = jnp.split(weight, self.groups, axis=-1)
            
            outputs = []
            for i in range(self.groups):
                group_out = lax.conv_transpose(
                    x_groups[i],
                    weight_groups[i],
                    strides=(self.stride, self.stride, self.stride),
                    padding=padding_lax,
                    dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
                    transpose_kernel=True
                )
                outputs.append(group_out)
            
            out = jnp.concatenate(outputs, axis=-1)
        
        if self.conv_transpose3d_bias is not None:
            out = out + self.conv_transpose3d_bias.reshape(1, 1, 1, 1, -1)
            
        # Convert back from NDHWC to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

# Test code
batch_size = 4
in_channels = 32  
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]