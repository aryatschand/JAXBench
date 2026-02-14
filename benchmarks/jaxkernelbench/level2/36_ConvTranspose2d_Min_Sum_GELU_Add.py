"""
JAXBench Level 2 - ConvTranspose2d_Min_Sum_GELU_Add
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)
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
        pad = kernel.shape[0] - 1 - self.padding
        padding = ((pad, pad), (pad, pad))
        
        out = jax.lax.conv_transpose(
            x_nhwc, 
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        # Handle output padding if needed
        if self.output_padding > 0:
            out = jax.lax.pad(
                out,
                padding_value=0.0,
                padding_config=((0, 0, 0), (0, self.output_padding, 0), 
                              (0, self.output_padding, 0), (0, 0, 0))
            )
            
        out = jnp.transpose(out, (0, 3, 1, 2))  # NHWC -> NCHW

        # Minimum operation along channel dimension
        out = jnp.min(out, axis=1, keepdims=True)
        
        # Sum operation along height dimension  
        out = jnp.sum(out, axis=2, keepdims=True)
        
        # GELU activation
        out = jax.nn.gelu(out)
        
        # Add bias
        out = out + self.bias
        
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(16, 64, 128, 128))]

def get_init_inputs():
    return [64, 128, 3, 2, 1, 1, (1, 1, 1)]