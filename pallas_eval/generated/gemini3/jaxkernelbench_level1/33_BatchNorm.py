"""
JAXBench Level 1 - Task 33: BatchNorm
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.132818
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bn_kernel(x_ref, scale_ref, offset_ref, o_ref):
    o_ref[...] = x_ref[...] * scale_ref[...] + offset_ref[...]

class Model:
    def __init__(self, num_features: int):
        self.num_features = num_features
        # Initialize with dummy values - will be overwritten by set_weights()
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.running_mean = jnp.zeros(num_features)
        self.running_var = jnp.ones(num_features)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, H, W = x.shape
        
        # Precompute scale and offset to fuse the batch norm operations
        scale = self.weight / jnp.sqrt(self.running_var + self.eps)
        offset = self.bias - self.running_mean * scale
        
        # Broadcast to match the reshaped x dimensions (N * C, 1)
        scale_2d = jnp.broadcast_to(scale, (N, C)).reshape(N * C, 1)
        offset_2d = jnp.broadcast_to(offset, (N, C)).reshape(N * C, 1)
        
        # Reshape x to 2D to avoid expensive transposes and simplify the kernel
        x_2d = x.reshape(N * C, H * W)
        
        def get_block_size(dim, preferred):
            for i in preferred:
                if dim % i == 0:
                    return i
            return 1
            
        block_nc = get_block_size(N * C, [8, 4, 2, 1])
        block_hw = get_block_size(H * W, [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
        
        grid_shape = (N * C // block_nc, H * W // block_hw)
        
        out_2d = pl.pallas_call(
            bn_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_nc, block_hw), lambda i, j: (i, j)),
                    pl.BlockSpec((block_nc, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_nc, 1), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_nc, block_hw), lambda i, j: (i, j)),
            ),
        )(x_2d, scale_2d, offset_2d)
        
        return out_2d.reshape(N, C, H, W)

batch_size = 64
features = 64
dim1 = 512 
dim2 = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
