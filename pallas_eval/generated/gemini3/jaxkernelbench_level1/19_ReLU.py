import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def relu_kernel(x_ref, o_ref):
    x = x_ref[...]
    zero = jnp.array(0, dtype=x.dtype)
    o_ref[...] = jnp.maximum(zero, x)

class Model:
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies ReLU activation to the input array.

        Args:
            x: Input JAX array of any shape.

        Returns:
            JAX array: Output array with ReLU applied, same shape as input.
        """
        orig_shape = x.shape
        
        if len(orig_shape) == 0:
            return jnp.maximum(jnp.array(0, dtype=x.dtype), x)
            
        if len(orig_shape) == 1:
            x_2d = x.reshape(orig_shape[0], 1)
        else:
            x_2d = x.reshape(orig_shape[0], -1)
            
        def get_block_dim(dim_size, preferred):
            for b in preferred:
                if dim_size % b == 0:
                    return b
            return 1
            
        b0 = get_block_dim(x_2d.shape[0], [128, 64, 32, 16, 8, 4, 2, 1])
        b1 = get_block_dim(x_2d.shape[1], [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
        
        block_shape = (b0, b1)
        grid_shape = (x_2d.shape[0] // b0, x_2d.shape[1] // b1)
        
        out_2d = pl.pallas_call(
            relu_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
            ),
        )(x_2d)
        
        return out_2d.reshape(orig_shape)

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
