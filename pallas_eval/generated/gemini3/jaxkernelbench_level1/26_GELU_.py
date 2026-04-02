import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gelu_kernel(x_ref, o_ref):
    o_ref[...] = jax.nn.gelu(x_ref[...])

class Model:
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        orig_shape = x.shape
        
        # Fast path for 2D tensors with compatible dimensions (like the benchmark shape)
        if len(orig_shape) == 2 and orig_shape[1] % 1024 == 0:
            if orig_shape[0] % 512 == 0:
                bs_0 = 512
            elif orig_shape[0] % 256 == 0:
                bs_0 = 256
            elif orig_shape[0] % 128 == 0:
                bs_0 = 128
            else:
                bs_0 = None
                
            if bs_0 is not None:
                grid_shape = (orig_shape[0] // bs_0, orig_shape[1] // 1024)
                return pl.pallas_call(
                    gelu_kernel,
                    out_shape=jax.ShapeDtypeStruct(orig_shape, x.dtype),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        grid=grid_shape,
                        in_specs=[pl.BlockSpec((bs_0, 1024), lambda i, j: (i, j))],
                        out_specs=pl.BlockSpec((bs_0, 1024), lambda i, j: (i, j)),
                    ),
                )(x)
                
        # Fallback for arbitrary shapes
        total_elements = x.size
        block_size = 128 * 128
        pad_len = (block_size - (total_elements % block_size)) % block_size
        
        if pad_len > 0:
            flat_x = jnp.pad(x.reshape(-1), (0, pad_len))
        else:
            flat_x = x.reshape(-1)
            
        flat_x = flat_x.reshape(-1, 128)
        grid_shape = (flat_x.shape[0] // 128,)
        
        out_flat = pl.pallas_call(
            gelu_kernel,
            out_shape=jax.ShapeDtypeStruct(flat_x.shape, flat_x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec((128, 128), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((128, 128), lambda i: (i, 0)),
            ),
        )(flat_x)
        
        if pad_len > 0:
            out_flat = out_flat.reshape(-1)[:-pad_len]
            
        return out_flat.reshape(orig_shape)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
