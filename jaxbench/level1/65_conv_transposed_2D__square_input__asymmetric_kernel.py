"""
JAXBench Level 1 - conv_transposed_2D__square_input__asymmetric_kernel
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        # Initialize weights with PyTorch shapes: (in_channels, out_channels, kH, kW)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size[0], kernel_size[1]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = (stride, stride)
        self.padding = ((kernel_size[0]-1-padding, kernel_size[0]-1-padding),
                       (kernel_size[1]-1-padding, kernel_size[1]-1-padding))
        self.output_padding = output_padding
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        if self.groups == 1:
            # Convert from NCHW to NHWC
            x = jnp.transpose(x, (0, 2, 3, 1))
            
            # Transpose kernel from (in, out, H, W) to (H, W, out, in)
            kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
            
            # Perform transposed convolution
            out = jax.lax.conv_transpose(
                x, kernel,
                strides=self.stride,
                padding=self.padding,
                dimension_numbers=('NHWC', 'HWOI', 'NHWC')
            )
            
            # Add bias if present
            if self.bias is not None:
                out = out + self.bias.reshape(1, 1, 1, -1)
                
            # Convert back from NHWC to NCHW
            out = jnp.transpose(out, (0, 3, 1, 2))
            
        else:
            # Handle grouped convolution
            x_groups = jnp.split(x, self.groups, axis=1)
            kernel_groups = jnp.split(self.weight, self.groups, axis=0)
            
            out_groups = []
            for x_group, kernel_group in zip(x_groups, kernel_groups):
                # Convert from NCHW to NHWC
                x_nhwc = jnp.transpose(x_group, (0, 2, 3, 1))
                
                # Transpose kernel from (in/groups, out/groups, H, W) to (H, W, out/groups, in/groups)
                kernel_hwoi = jnp.transpose(kernel_group, (2, 3, 1, 0))
                
                out_group = jax.lax.conv_transpose(
                    x_nhwc, kernel_hwoi,
                    strides=self.stride,
                    padding=self.padding,
                    dimension_numbers=('NHWC', 'HWOI', 'NHWC')
                )
                
                # Convert back from NHWC to NCHW
                out_group = jnp.transpose(out_group, (0, 3, 1, 2))
                out_groups.append(out_group)
                
            out = jnp.concatenate(out_groups, axis=1)
            
            if self.bias is not None:
                out = out + self.bias.reshape(1, -1, 1, 1)
                
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(8, 64, 512, 512))
    return [x]

def get_init_inputs():
    return [64, 64, (3, 7)]