"""
JAXBench Level 2 - ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros((1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        padding = ((self.kernel_size-1-self.padding,)*2,)*3  # For each spatial dim
        out = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(self.stride,)*3,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        out = jnp.transpose(out, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # Mean pooling over depth
        out = jnp.mean(out, axis=2, keepdims=True)

        # Add bias
        out = out + self.bias

        # Softmax over channels
        out = jax.nn.softmax(out, axis=1)

        # Tanh activation
        out = jnp.tanh(out)

        # Scale
        out = out * self.scaling_factor

        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (16, 16, 32, 128, 128))]

def get_init_inputs():
    return [16, 64, 3, 1, 1, 2.0]