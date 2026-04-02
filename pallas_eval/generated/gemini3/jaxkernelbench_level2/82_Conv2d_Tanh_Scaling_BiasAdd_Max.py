```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        # Initialize conv weights with same shape as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias_conv = jnp.zeros(out_channels)
        self.scaling_factor = scaling_factor
        self.bias = jnp.zeros(bias_shape)
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Prepare conv kernel
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        
        N, H_in, W_in, C = x.shape
        pool_size = self.pool_kernel_size
        
        H_out = H_in // pool_size
        W_out = W_in // pool_size
        
        # Slice x to perfectly match the pooled size (equivalent to VALID padding in pooling)
        x_sliced = x[:, :H_out * pool_size, :W_out * pool_size, :]
        
        # Find a suitable block size for the spatial dimensions
        def get_divisor(n, max_val=16):
            for i in range(max_val, 0, -1):
                if n % i == 0:
                    return i
            return 1
            
        block_H_out = get_divisor(H_out, 16)
        block_W_out = get_divisor(W_out, 16)
        
        block_H_in = block_H_out * pool_size
        block_W_in = block_W_out * pool_size
        
        grid_shape = (N, H_out // block_H_out, W_out // block_W_out)
        
        b_
