"""
JAXBench Level 1 - conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # ConvTranspose1d weight has shape (in_channels, out_channels, kernel_size)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size))
        if bias:
            self.bias = jnp.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCW -> NWC
        x = jnp.transpose(x, (0, 2, 1))
        
        # For dilated transposed convolution, we need to dilate the kernel
        # Transpose kernel (in, out, W) -> (W, out, in)
        kernel = jnp.transpose(self.weight, (2, 1, 0))
        
        # Dilate the kernel if dilation > 1
        if self.dilation > 1:
            kernel_w = kernel.shape[0]
            dilated_kernel_w = kernel_w + (kernel_w - 1) * (self.dilation - 1)
            dilated_kernel = jnp.zeros((dilated_kernel_w, kernel.shape[1], kernel.shape[2]), dtype=kernel.dtype)
            # Insert kernel values at dilated positions
            indices = jnp.arange(kernel_w) * self.dilation
            dilated_kernel = dilated_kernel.at[indices, :, :].set(kernel)
            kernel = dilated_kernel
            effective_kernel_size = dilated_kernel_w
        else:
            effective_kernel_size = self._kernel_size

        # Calculate padding for conv_transpose
        # For transposed conv: output_padding is handled by JAX automatically
        # padding in JAX conv_transpose is the amount to remove from output
        pad = effective_kernel_size - 1 - self.padding
        padding = ((pad, pad),)

        # ConvTranspose1d using conv_transpose (no dilation parameter - we dilated kernel manually)
        out = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride,),
            padding=padding,
            dimension_numbers=('NWC', 'WOI', 'NWC')
        )

        if self.bias_flag:
            out = out + self.bias[None, None, :]

        # Convert back NWC -> NCW
        out = jnp.transpose(out, (0, 2, 1))
        return out

    @property
    def kernel_size(self):
        return self.weight.shape[2]

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (16, 32, 131072))
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]