"""
JAXBench Level 1 - conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # Initialize weights with PyTorch shape (in_channels, out_channels, kH, kW)
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # For dilated transposed convolution, we need to use conv_general_dilated
        # with transposed=True or manually implement it
        
        # Transpose kernel from (in, out, H, W) to (H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # For transposed convolution with dilation, we need to compute the effective kernel size
        # effective_kernel_size = dilation * (kernel_size - 1) + 1
        effective_kernel_h = self.dilation * (self._kernel_size - 1) + 1
        effective_kernel_w = self.dilation * (self._kernel_size - 1) + 1
        
        # Calculate padding for conv_transpose
        # PyTorch output size: (input - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        # JAX padding calculation
        pad_h = effective_kernel_h - 1 - self.padding
        pad_w = effective_kernel_w - 1 - self.padding
        padding = ((pad_h, pad_h), (pad_w, pad_w))

        # Use conv_general_dilated with transposed dimensions
        # For transposed convolution, we swap lhs and rhs roles conceptually
        # But JAX's conv_transpose handles this
        
        # Dilate the kernel manually for transposed convolution
        if self.dilation > 1:
            # Insert zeros between kernel elements
            kh, kw, out_c, in_c = kernel.shape
            dilated_kh = (kh - 1) * self.dilation + 1
            dilated_kw = (kw - 1) * self.dilation + 1
            dilated_kernel = jnp.zeros((dilated_kh, dilated_kw, out_c, in_c))
            dilated_kernel = dilated_kernel.at[::self.dilation, ::self.dilation, :, :].set(kernel)
            kernel = dilated_kernel

        # Perform transposed convolution without dilation (since we dilated the kernel manually)
        out = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, -1)

        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

    @property
    def kernel_size(self):
        return self._kernel_size


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]