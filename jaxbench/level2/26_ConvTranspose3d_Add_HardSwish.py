"""
JAXBench Level 2 - ConvTranspose3d_Add_HardSwish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        # For ConvTranspose3d, weight shape is (in_channels, out_channels, D, H, W)
        self.conv_transpose_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x, add_input):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel (in, out, D, H, W) -> (D, H, W, out, in)
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))

        # For conv_transpose with output_padding, we need to handle it specially
        # PyTorch output size = (input - 1) * stride - 2 * padding + kernel_size + output_padding
        # JAX conv_transpose padding calculation
        pad_d = self.kernel_size - 1 - self.padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        
        # Add output_padding to the high side of padding
        padding = ((pad_d, pad_d + self.output_padding), 
                   (pad_h, pad_h + self.output_padding), 
                   (pad_w, pad_w + self.output_padding))

        # ConvTranspose3d
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        # Convert back NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # Add conv bias (shape: out_channels) - reshape for broadcasting
        x = x + self.conv_transpose_bias.reshape(1, -1, 1, 1, 1)

        # Add input tensor
        x = x + add_input

        # The original PyTorch code has: x = x * torch.nn.functional.hardswish(x)
        # This is x * hardswish(x), not just hardswish(x)
        # HardSwish: x * min(max(x + 3, 0), 6) / 6
        hardswish_x = x * jnp.minimum(jnp.maximum(x + 3, 0), 6) / 6
        x = x * hardswish_x

        return x


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    return [
        jax.random.uniform(key1, (batch_size, in_channels, D, H, W)),
        jax.random.uniform(key2, (batch_size, out_channels, D*stride, H*stride, W*stride))
    ]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]