import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def hardsigmoid_kernel(x_ref, o_ref):
    x = x_ref[...]
    # HardSigmoid: clip(clip(x + 3, 0) / 6, 1)
    val = jnp.maximum(x + 3.0, 0.0)
    val = val * (1.0 / 6.0)
    val = jnp.minimum(val, 1.0)
    o_ref[...] = val

class Model:
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies HardSigmoid activation to the input tensor.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with HardSigmoid applied, same shape as input.
        """
        if x.size == 0:
            return jnp.zeros_like(x)

        orig_shape = x.shape
        
        # Pallas requires at least 2D tensors
        if x.ndim == 0:
            x_2d = x.reshape(1, 1)
        elif x.ndim == 1:
            x_2d = x.reshape(x.shape[0], 1)
        elif x.ndim > 2:
            x_2d = x.reshape(x.shape[0], -1)
        else:
            x_2d = x
            
        # Determine block shape dynamically to ensure divisibility by powers of 2
        b0 = 512
        while b0 > 1 and x_2d.shape[0] % b0 != 0:
            b0 //= 2
            
        b1 = 1024
        while b1 > 1 and x_2d.shape[1] % b1 != 0:
            b1 //= 2
            
        block_shape = (b0, b1)
        grid_shape = (x_2d.shape[0] // b0, x_2d.shape[1] // b1)
        
        out_2d = pl.pallas_call(
            hardsigmoid_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
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
