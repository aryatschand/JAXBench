"""
JAXBench Level 2 - ConvTranspose2d_Mish_Add_Hardtanh_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        # Initialize weights with same shape as PyTorch ConvTranspose2d
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # (in,out,H,W) -> (H,W,out,in)
        
        # Calculate padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding), 
                  (pad_w, pad_w + self.output_padding))
        
        x = jax.lax.conv_transpose(
            x_nhwc, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, -1)
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Mish activation: x * tanh(softplus(x))
        x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
        
        # Add value
        x = x + self.add_value
        
        # Hardtanh
        x = jnp.clip(x, -1.0, 1.0)
        
        # Scale
        x = x * self.scale
        
        return x

    @property
    def kernel_size(self):
        return self.weight.shape[2]  # Assuming square kernel

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]