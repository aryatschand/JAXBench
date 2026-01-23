"""
JAXBench Level 1 - Task 64: conv_transposed_1D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.142219
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        # Initialize weights with same shapes as PyTorch
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels, kernel_size)  # PyTorch shape
        k = 1.0 / (in_channels * kernel_size)
        weight = jax.random.uniform(key, weight_shape) * (2.0 * jnp.sqrt(k)) - jnp.sqrt(k)
        
        # Transpose weight to JAX format (kernel_size, out_channels, in_channels) for transpose conv
        self.weight = jnp.transpose(weight, (2, 1, 0))
        
        if bias:
            self.bias = jax.random.uniform(key, (out_channels,)) * (2.0 * jnp.sqrt(k)) - jnp.sqrt(k)
        else:
            self.bias = None
            
        self.stride = stride
        self.pytorch_padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv1d_transpose.weight':
                # Convert PyTorch weight (in, out, kernel) to JAX (kernel, out, in) for transpose conv
                value = jnp.transpose(jnp.array(value), (2, 1, 0))
                setattr(self, 'weight', value)
            elif name == 'conv1d_transpose.bias':
                setattr(self, 'bias', jnp.array(value))

    def forward(self, x):
        # Convert NCL -> NLC for JAX
        x = jnp.transpose(x, (0, 2, 1))
        
        # Calculate padding for transposed convolution
        # For transpose conv: pad = kernel_size - 1 - pytorch_padding
        pad = self.kernel_size - 1 - self.pytorch_padding
        padding = ((pad, pad + self.output_padding),)
        
        # Perform transposed convolution using lax.conv_transpose
        # Note: lax.conv_transpose doesn't have feature_group_count, need to handle groups differently
        if self.groups == 1:
            out = lax.conv_transpose(
                x,
                self.weight,
                strides=(self.stride,),
                padding=padding,
                dimension_numbers=('NLC', 'LOI', 'NLC'),
                transpose_kernel=True
            )
        else:
            # Handle grouped convolution manually by splitting and concatenating
            batch_size = x.shape[0]
            length = x.shape[1]
            in_channels = x.shape[2]
            out_channels = self.weight.shape[1]
            
            in_channels_per_group = in_channels // self.groups
            out_channels_per_group = out_channels // self.groups
            
            outputs = []
            for g in range(self.groups):
                x_group = x[:, :, g * in_channels_per_group:(g + 1) * in_channels_per_group]
                weight_group = self.weight[:, g * out_channels_per_group:(g + 1) * out_channels_per_group, :]
                
                out_group = lax.conv_transpose(
                    x_group,
                    weight_group,
                    strides=(self.stride,),
                    padding=padding,
                    dimension_numbers=('NLC', 'LOI', 'NLC'),
                    transpose_kernel=True
                )
                outputs.append(out_group)
            
            out = jnp.concatenate(outputs, axis=2)
        
        if self.bias is not None:
            out = out + self.bias
            
        # Convert back NLC -> NCL
        out = jnp.transpose(out, (0, 2, 1))
        return out

# Test code
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
length = 65536

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]