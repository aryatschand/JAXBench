"""
JAXBench Level 2 - Conv2d_Mish_Mish
Translated from KernelBench PyTorch to JAX using bedrock/opus.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax


class Model:
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights with same shapes as PyTorch Conv2d
        # PyTorch Conv2d weight shape: (out_channels, in_channels, kernel_height, kernel_width)
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        
        # Kaiming uniform initialization (similar to PyTorch default)
        fan_in = in_channels * kernel_size * kernel_size
        bound = jnp.sqrt(1.0 / fan_in)
        
        self.conv_weight = jax.random.uniform(key1, (out_channels, in_channels, kernel_size, kernel_size), 
                                         minval=-bound, maxval=bound)
        self.conv_bias = jax.random.uniform(key2, (out_channels,), minval=-bound, maxval=bound)
    
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Perform convolution using jax.lax.conv_general_dilated
        # Input x shape: (batch, in_channels, height, width) - NCHW format
        # Weight shape: (out_channels, in_channels, kernel_height, kernel_width) - OIHW format
        
        x = lax.conv_general_dilated(
            x,
            self.conv_weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        # Add bias - reshape bias to broadcast correctly
        x = x + self.conv_bias.reshape(1, -1, 1, 1)
        
        # Apply Mish activation: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        x = x * jnp.tanh(jax.nn.softplus(x))
        
        # Apply Mish activation again
        x = x * jnp.tanh(jax.nn.softplus(x))
        
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]