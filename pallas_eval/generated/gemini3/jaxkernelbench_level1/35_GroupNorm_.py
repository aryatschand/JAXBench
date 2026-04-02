"""
JAXBench Level 1 - Task 35: GroupNorm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.133536
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def group_norm_fwd_kernel(x_ref, mean_ref, rstd_ref, weight_ref, bias_ref, o_ref):
    x = x_ref[...]
    mean = mean_ref[...]
    rstd = rstd_ref[...]
    weight = weight_ref[...]
    bias = bias_ref[...]
    
    o_ref[...] = (x - mean) * rstd * weight + bias

class Model:
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        self.num_groups = num_groups
        self.num_features = num_features
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Applies Group Normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_features, *).

        Returns:
            Output tensor with Group Normalization applied, same shape as input.
        """
        input_shape = x.shape
        batch_size = input_shape[0]
        num_channels = input_shape[1]
        
        spatial_shape = input_shape[2:]
        spatial_size = 1
        for d in spatial_shape:
            spatial_size *= d
            
        # Calculate mean and var using JAX native reductions
        x_group = x.reshape((batch_size * self.num_groups, (num_channels // self.num_groups) * spatial_size))
        mean = jnp.mean(x_group, axis=1, keepdims=True)
        var = jnp.var(x_group, axis=1, keepdims=True)
        
        # Prepare inputs for Pallas kernel
        mean = jnp.asarray(mean, dtype=x.dtype)
        rstd = jnp.asarray(1.0 / jnp.sqrt(var + self.eps), dtype=x.dtype)
        
        mean = jnp.repeat(mean, num_channels // self.num_groups, axis=0)
        rstd = jnp.repeat(rstd, num_channels // self.num_groups, axis=0)
        
        weight = jnp.tile(jnp.asarray(self.weight, dtype=x.dtype), batch_size).reshape(-1, 1)
        bias = jnp.tile(jnp.asarray(self.bias, dtype=x.dtype), batch_size).reshape(-1, 1)
        
        x_2d = x.reshape((batch_size * num_channels, spatial_size))
        
        # Pad spatial dimension to be a multiple of 128 for optimal TPU performance
        pad_n = 0
        if spatial_size % 128 != 0:
            pad_n = 128 - (spatial_size % 128)
            x_2d = jnp.pad(x_2d, ((0, 0), (0, pad_n)))
            
        padded_spatial_size = spatial_size + pad_n
        
        # Determine block sizes
        block_m = min(batch_size * num_channels, 16)
        while (batch_size * num_channels) % block_m != 0:
            block_m -= 1
            
        block_n = min(padded_spatial_size, 1024)
        while padded_spatial_size % block_n != 0:
            block_n -= 128
            if block_n <= 128:
                block_n = 128
                break
                
        grid = ((batch_size * num_channels) // block_m, padded_spatial_size // block_n)
        
        out_2d = pl.pallas_call(
            group_norm_fwd_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            )
        )(x_2d, mean, rstd, weight, bias)
        
        if pad_n > 0:
            out_2d = out_2d[:, :-pad_n]
            
        return out_2d.reshape(input_shape)

batch_size = 16  # Reduced from 112 for memory
features = 64
num_groups = 8
dim1 = 256  # Reduced from 512
dim2 = 256  # Reduced from 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features, num_groups]
