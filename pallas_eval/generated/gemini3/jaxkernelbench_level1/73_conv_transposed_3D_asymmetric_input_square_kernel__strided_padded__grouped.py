```python
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bias_transpose_kernel(x_ref, bias_ref, o_ref):
    # x_ref shape: (1, D_block, H_block, W_block, C_block)
    # bias_ref shape: (C_block,)
    # o_ref shape: (1, C_block, D_block, H_block, W_block)
    
    x = x_ref[0, :, :, :, :]
    bias = bias_ref[:]
    
    # Add bias
    res = x + bias[None, None, None, :]
    
    # Transpose from (D, H, W, C) to (C, D, H, W)
    o_ref[0, :, :, :, :] = jnp.transpose(res, (3, 0, 1, 2))

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights with correct shape for ConvTranspose3d
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.conv_transpose3d_weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.conv_transpose3d_bias = jnp.zeros(out_channels)
        else:
            self.conv_transpose3d_bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose weight from (in, out, D, H, W) to (D, H, W, out, in)
        weight = jnp.transpose(self.conv
