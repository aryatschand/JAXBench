"""
JAXBench Level 2 - ConvTranspose2d_Add_Min_GELU_Multiply
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def pallas_elementwise_fused(x, bias, add_value, multiply_value):
    N, H, W, C = x.shape
    M = N * H * W
    x_flat = x.reshape((M, C))
    
    block_M = 512
    while M % block_M != 0 and block_M > 1:
        block_M //= 2
        
    block_C = 128
    while C % block_C != 0 and block_C > 1:
        block_C //= 2
        
    grid_shape = (M // block_M, C // block_C)
    
    add_val_arr = jnp.asarray(add_value, dtype=x.dtype).reshape((1,))
    mul_val_arr = jnp.asarray(multiply_value, dtype=x.dtype).reshape((1,))
    
    def elementwise_fused_kernel(x_ref, bias_ref, add_ref, mul_ref, o_ref):
        x_val = x_ref[:, :]
        bias_val = bias_ref[:]
        add_val = add_ref[0]
        mul_val = mul_ref[0]
        
        # Add bias
        x_val = x_val + bias_val.reshape(1, -1)
        
        # Add add_value
        x_val = x_val + add_val
        
        # Minimum with 0.0
        x_val = jnp.minimum(x_val, 0.0)
        
        # GELU
        x_val = gelu(x_val)
        
        # Multiply
        x_val = x_val * mul_val
        
        o_ref[:, :] = x_val

    out_flat = pl.pallas_call(
        elementwise_fused_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((block_M, block_C), lambda i, j: (i, j)),
                pl.BlockSpec((block_C,), lambda i, j: (j,)),
                pl.BlockSpec((1,), lambda i, j: (0,)),
                pl.BlockSpec((1,), lambda i, j: (0,)),
            ],
            out_specs=pl.BlockSpec((block_M, block_C), lambda i, j: (i, j)),
        )
    )(x_flat, bias, add_val_arr, mul_val_arr)
    
    return out_flat.reshape((N, H, W, C))

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        # Initialize with PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, k, k)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.add_value = add_value
        self.multiply_value = multiply_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel (in,out,H,W) -> (H,W,out,in) for JAX conv_transpose
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Calculate padding (kernel_size - 1 - pytorch_padding)
        padding = ((self.kernel_size - 1, self.kernel_size - 1),
                  (self.kernel_size - 1, self.kernel_size - 1))
        
        # Transposed convolution
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        # Fused elementwise operations in NHWC
        x = pallas_elementwise_fused(x, self.bias, self.add_value, self.multiply_value)
        
        # Convert back NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
