"""
JAXBench Level 2 - ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        # ConvTranspose3d weight shape: (in_channels, out_channels, kD, kH, kW)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.scale1 = jnp.array(scale1)
        self.bias = jnp.zeros(bias_shape)
        self.scale2 = jnp.array(scale2)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # ConvTranspose3d
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in, out, D, H, W) -> (D, H, W, out, in)
        
        # For conv_transpose, padding calculation:
        # JAX padding = kernel_size - 1 - pytorch_padding
        pad_val = self.kernel_size - 1 - self.padding
        padding = ((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val))
        
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        # Add conv bias
        x = x + self.conv_transpose_bias
        
        # Scale1
        x = x * self.scale1
        
        # AvgPool3d with kernel_size=2
        x = jax.lax.reduce_window(
            x,
            init_value=0.,
            computation=jax.lax.add,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        ) / 8.0  # Divide by window size (2*2*2)
        
        # Add bias - bias shape is (out_channels, 1, 1, 1) in PyTorch (NCDHW format)
        # In NDHWC format, we need to reshape bias to (1, 1, 1, 1, out_channels)
        bias_reshaped = self.bias.reshape(1, 1, 1, 1, -1)
        x = x + bias_reshaped
        
        # Scale2
        x = x * self.scale2
        
        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(128, 3, 16, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, 0.5, 1.0, (16, 1, 1, 1)]