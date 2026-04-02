```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import leaky_relu

def epilogue_kernel(x_ref, bias_ref, divisor_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    divisor = divisor_ref[...]
    
    # Broadcasting works natively for (block_c, block_n) + (block_c, 1)
    x = x + bias
    x = x / divisor
    
    # LeakyReLU
    x = jnp.where(x >= 0.0, x, x * 0.01)
    
    o_ref[...] = x

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel for JAX conv
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Perform convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        B, H, W, C = x.shape
        
        # Transpose to (C, B, H, W) and flatten to (C, N) to align with TPU block requirements
        x = jnp.transpose(x, (3, 0, 1, 2))
        N = B * H * W
        x_flat = x.reshape(C, N)
        
        # Reshape bias and divisor to 2D for Pallas
        bias_flat = self.bias.reshape(C, 1)
        divisor_arr = jnp.array(self.divisor, dtype=x.dtype).reshape(1, 1)
        
        # Block dimensions: multiples of (8, 128)
        block_c = 64
        block_n = 512
