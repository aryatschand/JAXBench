```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import gelu, leaky_relu

def fused_kernel(x_ref, bias_ref, sum_t_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    sum_t = sum_t_ref[...]
    
    # Broadcast bias and sum_t to match x's block shape
    bias = bias.reshape(1, 128)
    bias = pltpu.repeat(bias, x.shape[0], axis=0)
    
    sum_t = sum_t.reshape(1, 128)
    sum_t = pltpu.repeat(sum_t, x.shape[0], axis=0)
    
    # Fused operations
    x = x + bias
    x = jnp.where(x >= 0.0, x, x * 0.2)  # leaky_relu
    x = x + sum_t
    x = jnp.clip(x, -1.0, 1.0)
    x = gelu(x)
    
    o_ref[...] = x

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.sum_tensor = jnp.zeros(sum_tensor_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (out,in,D,H,W) to (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        N, D, H, W, C = x.shape
        M =
