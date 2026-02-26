"""
JAXBench Level 2 - ConvTranspose3d_Clamp_Min_Divide
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        # For ConvTranspose3d, weight shape is (in_channels, out_channels, D, H, W)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.pytorch_padding = padding
        self.min_value = min_value
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Calculate padding: for conv_transpose, pad = kernel_size - 1 - pytorch_padding
        pad_val = self.kernel_size - 1 - self.pytorch_padding
        padding = ((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val))
        
        # Apply transposed convolution
        x = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back from NDHWC to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Apply clamp and division
        x = jnp.maximum(x, self.min_value)
        x = x / self.divisor
        
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]