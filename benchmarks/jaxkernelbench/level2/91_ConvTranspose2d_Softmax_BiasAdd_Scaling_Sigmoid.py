"""
JAXBench Level 2 - ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import softmax, sigmoid

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (in, out, H, W) to (H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Calculate padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding), 
                  (pad_w, pad_w + self.output_padding))
        
        # Transposed convolution
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Apply operations
        x = softmax(x, axis=1)
        x = x + self.bias
        x = x * self.scaling_factor
        x = sigmoid(x)
        
        return x

    @property
    def kernel_size(self):
        return self.weight.shape[2]  # Assuming square kernel

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]