"""
JAXBench Level 1 - conv_transposed_3D__square_input__square_kernel
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        # Initialize weights with PyTorch shape (in_channels, out_channels, D, H, W)
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        # Calculate padding
        pad_d = self.kernel_size - 1 - self.padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_d, pad_d + self.output_padding),
                  (pad_h, pad_h + self.output_padding),
                  (pad_w, pad_w + self.output_padding))

        if self.groups == 1:
            out = jax.lax.conv_transpose(
                x, kernel,
                strides=(self.stride, self.stride, self.stride),
                padding=padding,
                dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
        else:
            # Handle grouped convolution
            x_groups = jnp.split(x, self.groups, axis=-1)
            k_groups = jnp.split(kernel, self.groups, axis=-1)
            out_groups = []
            
            for x_group, k_group in zip(x_groups, k_groups):
                out_group = jax.lax.conv_transpose(
                    x_group, k_group,
                    strides=(self.stride, self.stride, self.stride),
                    padding=padding,
                    dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
                out_groups.append(out_group)
            
            out = jnp.concatenate(out_groups, axis=-1)

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)

        # Convert back from NDHWC to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

    @property
    def kernel_size(self):
        return self.weight.shape[2]  # Assuming square kernel

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(8, 48, 64, 64, 64))
    return [x]

def get_init_inputs():
    return [48, 48, 3]  # in_channels, out_channels, kernel_size