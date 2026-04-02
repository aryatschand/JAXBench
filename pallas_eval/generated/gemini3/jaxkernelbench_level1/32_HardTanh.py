import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def hardtanh_kernel(x_ref, o_ref):
    o_ref[...] = jnp.clip(x_ref[...], -1.0, 1.0)

def get_block_size(dim, max_size=1024):
    for p in [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        if dim % p == 0 and p <= dim:
            return p
    return 1

class Model:
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 0:
            x_2d = x.reshape((1, 1))
        elif len(orig_shape) == 1:
            x_2d = x.reshape((orig_shape[0], 1))
        elif len(orig_shape) > 2:
            x_2d = x.reshape((orig_shape[0], -1))
        else:
            x_2d = x
            
        block_0 = get_block_size(x_2d.shape[0], 1024)
        block_1 = get_block_size(x_2d.shape[1], 1024)
        
        if block_0 == 1 or block_1 == 1:
            return jnp.clip(x, -1.0, 1.0)
            
        grid_shape = (x_2d.shape[0] // block_0, x_2d.shape[1] // block_1)
        
        out_2d = pl.pallas_call(
            hardtanh_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec((block_0, block_1), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_0, block_1), lambda i, j: (i, j)),
            ),
        )(x_2d)
        
        return out_2d.reshape(orig_shape)

    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
