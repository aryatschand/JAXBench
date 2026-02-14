"""
JAXBench Level 1 - conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # PyTorch weight shape: (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias = jnp.zeros(out_channels)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # For dilated transposed convolution, we need to use conv_general_dilated
        # with the kernel flipped and appropriate padding calculation
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Flip the kernel for transposed convolution
        kernel = jnp.flip(kernel, axis=(0, 1, 2))
        
        # Calculate effective kernel size with dilation
        effective_kernel_size = self.dilation * (self.kernel_size - 1) + 1
        
        # For transposed convolution, we need to calculate output padding
        # PyTorch ConvTranspose3d output size: (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
        
        # Calculate padding for conv_general_dilated to simulate transposed convolution
        # We need to pad the input and use stride=1 on the dilated input
        
        # Dilate the input by inserting zeros
        batch_size, d_in, h_in, w_in, channels = x.shape
        
        if self.stride > 1:
            # Insert zeros between input elements
            d_dilated = d_in + (d_in - 1) * (self.stride - 1)
            h_dilated = h_in + (h_in - 1) * (self.stride - 1)
            w_dilated = w_in + (w_in - 1) * (self.stride - 1)
            
            x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
            x_dilated = x_dilated.at[:, ::self.stride, ::self.stride, ::self.stride, :].set(x)
        else:
            x_dilated = x
            d_dilated, h_dilated, w_dilated = d_in, h_in, w_in
        
        # Calculate padding for the convolution
        # Total padding needed on each side
        pad_d = effective_kernel_size - 1 - self.padding
        pad_h = effective_kernel_size - 1 - self.padding
        pad_w = effective_kernel_size - 1 - self.padding
        
        padding = ((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w))
        
        # Perform regular convolution with dilated kernel
        out = jax.lax.conv_general_dilated(
            x_dilated,
            kernel,
            window_strides=(1, 1, 1),
            padding=padding,
            lhs_dilation=(1, 1, 1),
            rhs_dilation=(self.dilation, self.dilation, self.dilation),
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)

        # Convert back from NDHWC to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (16, 32, 16, 32, 32))
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]