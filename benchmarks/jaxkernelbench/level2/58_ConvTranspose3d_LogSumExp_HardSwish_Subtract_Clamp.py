"""
JAXBench Level 2 - ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        # ConvTranspose3d weight shape: (in_channels, out_channels, D, H, W)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel (in, out, D, H, W) -> (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Calculate output padding
        kernel_size = self.weight.shape[2]
        pad_size = kernel_size - 1 - 1  # kernel_size - 1 - pytorch_padding
        padding = [(pad_size, pad_size)] * 3
        
        # ConvTranspose3d
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(2, 2, 2),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        # Convert back NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # LogSumExp
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
        
        # HardSwish: x * sigmoid(x+3)/6
        x = x * jax.nn.sigmoid(x + 3) / 6
        
        # Subtract bias
        x = x - self.bias
        
        # Clamp
        x = jnp.clip(x, -1, 1)
        
        return x

batch_size = 128
in_channels = 3  
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]