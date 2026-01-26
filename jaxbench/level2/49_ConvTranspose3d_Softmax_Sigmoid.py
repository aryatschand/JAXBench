"""
JAXBench Level 2 - ConvTranspose3d_Softmax_Sigmoid
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        # For ConvTranspose3d, PyTorch weight shape is (in_channels, out_channels, D, H, W)
        k = kernel_size
        if isinstance(k, int):
            k = (k, k, k)
        self.kernel_size = k  # Store kernel_size as instance attribute
        self.weight = jnp.zeros((in_channels, out_channels, k[0], k[1], k[2]))
        if bias:
            self.bias = jnp.zeros((out_channels,))
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.has_bias = bias

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Calculate padding
        pad_d = self.kernel_size[0] - 1 - self.padding[0]
        pad_h = self.kernel_size[1] - 1 - self.padding[1]
        pad_w = self.kernel_size[2] - 1 - self.padding[2]
        padding = ((pad_d, pad_d + self.output_padding[0]),
                  (pad_h, pad_h + self.output_padding[1]),
                  (pad_w, pad_w + self.output_padding[2]))

        x = jax.lax.conv_transpose(
            x, kernel,
            strides=self.stride,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))

        if self.has_bias:
            x = x + self.bias.reshape(1, 1, 1, 1, -1)

        # Convert back from NDHWC to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Apply softmax along channel dimension (dim=1)
        x = jnn.softmax(x, axis=1)
        
        # Apply sigmoid
        x = jnn.sigmoid(x)
        
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]