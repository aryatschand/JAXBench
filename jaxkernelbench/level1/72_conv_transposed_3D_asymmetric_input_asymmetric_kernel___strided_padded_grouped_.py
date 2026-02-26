"""
JAXBench Level 1 - conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        # Initialize weights with PyTorch shape (in_channels, out_channels/groups, D, H, W)
        self.weight = jnp.zeros((in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.pytorch_padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.groups = groups
        self.out_channels = out_channels
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Calculate JAX padding: kernel_size - 1 - pytorch_padding
        jax_padding = tuple(k - 1 - p for k, p in zip(self.kernel_size, self.pytorch_padding))
        
        if self.groups == 1:
            # Convert NCDHW -> NDHWC
            x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
            
            # Transpose kernel from (in, out, D, H, W) -> (D, H, W, out, in)
            kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
            
            out = jax.lax.conv_transpose(
                x_ndhwc, kernel,
                strides=self.stride,
                padding=((jax_padding[0], jax_padding[0] + self.output_padding[0]), 
                        (jax_padding[1], jax_padding[1] + self.output_padding[1]),
                        (jax_padding[2], jax_padding[2] + self.output_padding[2])),
                dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
            
            # Convert back NDHWC -> NCDHW
            out = jnp.transpose(out, (0, 4, 1, 2, 3))
            
        else:
            # Handle grouped convolution
            # For grouped ConvTranspose3d:
            # - Input has in_channels total, split into groups
            # - Weight shape is (in_channels, out_channels/groups, D, H, W)
            # - Each group processes in_channels/groups input channels
            # - Each group produces out_channels/groups output channels
            
            in_per_group = x.shape[1] // self.groups
            out_per_group = self.out_channels // self.groups
            
            # Split input into groups along channel dimension
            x_groups = jnp.split(x, self.groups, axis=1)
            # Split weights - weight shape is (in_channels, out_channels/groups, D, H, W)
            # Each group uses in_channels/groups of the input channels
            w_groups = jnp.split(self.weight, self.groups, axis=0)
            
            # Process each group separately
            out_groups = []
            for x_g, w_g in zip(x_groups, w_groups):
                # Convert NCDHW -> NDHWC
                x_g_ndhwc = jnp.transpose(x_g, (0, 2, 3, 4, 1))
                
                # Transpose kernel from (in_per_group, out_per_group, D, H, W) -> (D, H, W, out_per_group, in_per_group)
                kernel = jnp.transpose(w_g, (2, 3, 4, 1, 0))
                
                out_g = jax.lax.conv_transpose(
                    x_g_ndhwc, kernel,
                    strides=self.stride,
                    padding=((jax_padding[0], jax_padding[0] + self.output_padding[0]),
                            (jax_padding[1], jax_padding[1] + self.output_padding[1]),
                            (jax_padding[2], jax_padding[2] + self.output_padding[2])),
                    dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
                
                # Convert back NDHWC -> NCDHW
                out_g = jnp.transpose(out_g, (0, 4, 1, 2, 3))
                out_groups.append(out_g)
                
            out = jnp.concatenate(out_groups, axis=1)
            
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)
            
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 32, 12, 24, 48))
    return [x]

def get_init_inputs():
    return [32, 32, (3, 5, 7), (2, 2, 2), (1, 2, 3), (1, 1, 1), 4]