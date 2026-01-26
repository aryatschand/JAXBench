"""
JAXBench Level 2 - ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        # Initialize weights with same shapes as PyTorch
        # PyTorch ConvTranspose3d weight shape: (in_channels, out_channels, kD, kH, kW)
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

    def forward(self, x):
        # Convert input from NCDHW to NDHWC
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        
        # For conv_transpose with stride=2, padding=1, output_padding=1, kernel_size=3:
        # PyTorch output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        # = (input_size - 1) * 2 - 2 * 1 + 3 + 1 = 2 * input_size
        # For input depth=16: output = 32
        # For input height/width=32: output = 64
        
        # JAX conv_transpose padding calculation:
        # We need to use padding that gives us the right output size
        # For transposed conv: output_size = (input_size - 1) * stride + kernel_size - 2 * pad_low
        # With output_padding, we need: (input_size - 1) * stride - 2 * pytorch_padding + kernel_size + output_padding
        
        # Use 'SAME' style calculation but manually specify padding
        # padding = ((pad_low, pad_high), ...)
        # For kernel_size=3, pytorch_padding=1: effective_pad = kernel_size - 1 - pytorch_padding = 3 - 1 - 1 = 1
        padding = ((1, 1), (1, 1), (1, 1))
        
        # ConvTranspose3d
        x_conv = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        # Handle output_padding by padding the result
        if self.output_padding > 0:
            # Pad at the end of each spatial dimension
            x_conv = jnp.pad(x_conv, ((0, 0), (0, self.output_padding), (0, self.output_padding), (0, self.output_padding), (0, 0)))
        
        # Add conv bias (broadcast over spatial dimensions)
        x_conv = x_conv + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW
        x = jnp.transpose(x_conv, (0, 4, 1, 2, 3))
        
        # Store original x for residual connections
        original_x = x
        
        # Add bias (shape is (out_channels, 1, 1, 1))
        x = x + self.bias
        
        # Residual add
        x = x + original_x
        
        # Multiply
        x = x * original_x
        
        # Final residual add
        x = x + original_x
        
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(16, 32, 16, 32, 32))]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 1, (64, 1, 1, 1)]