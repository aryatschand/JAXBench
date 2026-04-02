import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def softsign_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x / (1.0 + jnp.abs(x))

def get_block_dim(dim_size, max_block):
    b = max_block
    while b > 1:
        if dim_size % b == 0:
            return b
        b //= 2
    return 1

class Model:
    """
    Simple model that performs a Softsign activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1)
        
        if x_flat.size % 1024 == 0:
            x_2d = x_flat.reshape(-1, 1024)
            b0 = get_block_dim(x_2d.shape[0], 256)
            block = (b0, 1024)
        elif x_flat.size % 128 == 0:
            x_2d = x_flat.reshape(-1, 128)
            b0 = get_block_dim(x_2d.shape[0], 256)
            block = (b0, 128)
        else:
            x_2d = x_flat.reshape(1, -1)
            b1 = get_block_dim(x_2d.shape[1], 1024)
            block = (1, b1)
            
        grid_shape = (x_2d.shape[0] // block[0], x_2d.shape[1] // block[1])
        
        out_2d = pl.pallas_call(
            softsign_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x_2d)
        
        return out_2d.reshape(orig_shape)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
