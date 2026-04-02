"""
JAXBench Level 1 - conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
Optimized with Pallas TPU kernel for bias addition and native XLA lhs_dilation for massive speedup.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_bias_kernel(x_ref, bias_ref, o_ref):
    x_val = x_ref[...]
    bias_val = bias_ref[...]  # shape is (1, 128)
    o_ref[...] = x_val + bias_val

def identity_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

def pallas_add_bias(x, bias):
    orig_shape = x.shape
    C = orig_shape[-1]
    x_2d = x.reshape(-1, C)
    M = x_2d.shape[0]
    
    block_m = 128
    pad_m = (block_m - (M % block_m)) % block_m
    pad_c = (128 - (C % 128)) % 128  # Pad C to multiple of 128 for best TPU performance
    
    if pad_m > 0 or pad_c > 0:
        x_2d = jnp.pad(x_2d, ((0, pad_m), (0, pad_c)))
    
    padded_C = x_2d.shape[1]
    
    if bias is not None:
        if pad_c > 0:
            bias_padded = jnp.pad(bias, (0, pad_c))
        else:
            bias_padded = bias
            
        # Reshape to 2D to satisfy Pallas constraints
        bias_padded = bias_padded.reshape(1, -1)
            
        grid = (x_2d.shape[0] // block_m, padded_C // 128)
        
        out_2d = pl.pallas_call(
            add_bias_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, 128), lambda i, j: (i, j)),
                    pl.BlockSpec((1, 128), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, 128), lambda i, j: (i, j)),
            )
        )(x_2d, bias_padded)
    else:
        grid = (x_2d.shape[0] // block_m, padded_C // 128)
        out_2d = pl.pallas_call(
            identity_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, 128), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, 128), lambda i, j: (i, j)),
            )
        )(x_2d)
        
    if pad_m > 0 or pad_c > 0:
        out_2d = out_2d[:M, :C]
        
    return out_2d.reshape(orig_shape)

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # PyTorch weight shape: (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias = jnp.zeros(out_channels)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Flip the kernel for transposed convolution
        kernel = jnp.flip(kernel, axis=(0, 1, 2))
        
        # Calculate effective kernel size with dilation
        effective_kernel_size = self.dilation * (self.kernel_size - 1) + 1
        
        # Calculate padding for the convolution
        pad_d = effective_kernel_size - 1 - self.padding
        pad_h = effective_kernel_size - 1 - self.padding
        pad_w = effective_kernel_size - 1 - self.padding
        
        padding = ((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w))
        
        # Perform regular convolution with dilated kernel
        # Using lhs_dilation avoids the extremely slow manual padding
        out = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding=padding,
            lhs_dilation=(self.stride, self.stride, self.stride),
            rhs_dilation=(self.dilation, self.dilation, self.dilation),
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        # Add bias if present using Pallas kernel
        out = pallas_add_bias(out, self.bias)

        # Convert back from NDHWC to NCDHW
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (16, 32, 16, 32, 32))
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]
