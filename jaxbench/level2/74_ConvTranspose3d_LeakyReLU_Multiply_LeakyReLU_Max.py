"""
JAXBench Level 2 - ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        
        kernel_size = self.weight.shape[2]
        
        # For conv_transpose with stride > 1 and output_padding, we need to handle it carefully
        # PyTorch output size = (input - 1) * stride - 2 * padding + kernel_size + output_padding
        # JAX conv_transpose padding calculation is different
        
        # Use 'SAME' style calculation but manually specify padding
        # For transposed conv: we need padding = kernel_size - 1 - pytorch_padding on each side
        pad_amount = kernel_size - 1 - self.padding
        padding = ((pad_amount, pad_amount + self.output_padding), 
                   (pad_amount, pad_amount + self.output_padding), 
                   (pad_amount, pad_amount + self.output_padding))
        
        out = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)
        out = jnp.transpose(out, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # LeakyReLU
        out = jnp.where(out > 0, out, out * 0.2)

        # Multiply by learnable parameter
        out = out * self.multiplier

        # LeakyReLU
        out = jnp.where(out > 0, out, out * 0.2)

        # MaxPool3d with kernel_size=2, stride=2
        out = jnp.transpose(out, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        out = jax.lax.reduce_window(
            out,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        )
        out = jnp.transpose(out, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
        
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(16, 16, 16, 32, 32))]

def get_init_inputs():
    return [16, 32, 3, 2, 1, 1, (32, 1, 1, 1)]