```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax

def transpose_bias_kernel(x_ref, b_ref, o_ref):
    x = x_ref[...]
    b = b_ref[...]
    # x is (1, block_H, block_W, block_C), b is (block_C,)
    x = x + b.reshape(1, 1, 1, -1)
    o_ref[...] = jnp.transpose(x, (0, 3, 1, 2))

def transpose_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.transpose(x, (0, 3, 1, 2))

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        self.bias = None
        if bias:
            self.bias = jnp.zeros(out_channels)
            
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = ((kernel_size[0]-1-padding[0], kernel_size[0]-1-padding[0]),
                       (kernel_size[1]-1-padding[1], kernel_size[1]-1-padding[1]))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv_transpose2d.weight':
                # Convert PyTorch weight (in_channels, out_channels, kH, kW) to 
                # JAX format (kH, kW, out_channels, in_channels)
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
            elif name == 'conv_transpose2d.bias':
                value = jnp.array(value)
            setattr(self, name.replace('conv_transpose2d.', ''), jnp.array(value))

    def forward(self, x):
        # Perform transposed convolution using lax.conv_transpose
        # Input is NCHW, weight is HWOI, output is NHWC
        # This avoids the expensive NCHW -> NHWC input transpose
        out =
