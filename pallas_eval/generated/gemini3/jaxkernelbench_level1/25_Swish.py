import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def swish_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x * jax.nn.sigmoid(x)

class Model:
    """
    Simple model that performs a Swish activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Swish activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with Swish applied, same shape as input.
        """
        orig_shape = x.shape
        
        if len(orig_shape) == 1:
            x_2d = x.reshape(orig_shape[0], 1)
        elif len(orig_shape) == 0:
            x_2d = x.reshape(1, 1)
        else:
            x_2d = x.reshape(orig_shape[0], -1)
            
        bs0 = min(x_2d.shape[0], 512)
        bs1 = min(x_2d.shape[1], 1024)
        
        if x_2d.shape[0] % bs0 != 0 or x_2d.shape[1] % bs1 != 0:
            return x * jax.nn.sigmoid(x)
            
        grid_shape = (x_2d.shape[0] // bs0, x_2d.shape[1] // bs1)
        
        out = pl.pallas_call(
            swish_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec((bs0, bs1), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((bs0, bs1), lambda i, j: (i, j)),
            ),
        )(x_2d)
        
        return out.reshape(orig_shape)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
