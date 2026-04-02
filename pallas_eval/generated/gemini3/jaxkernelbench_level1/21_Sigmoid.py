import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def sigmoid_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jax.nn.sigmoid(x)

class Model:
    """
    Simple model that performs a Sigmoid activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Sigmoid activation to the input tensor.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with Sigmoid applied, same shape as input.
        """
        if x.size == 0:
            return x
            
        # Flatten and reshape to 2D to satisfy TPU Pallas constraints
        # We aim for an inner dimension that is a multiple of 128 (preferably 1024)
        # and an outer dimension that is a multiple of 8.
        if x.size % 1024 == 0:
            x_2d = x.reshape(-1, 1024)
            if x_2d.shape[0] % 512 == 0:
                block_0 = 512
            elif x_2d.shape[0] % 128 == 0:
                block_0 = 128
            elif x_2d.shape[0] % 8 == 0:
                block_0 = 8
            else:
                return jax.nn.sigmoid(x)
            block_shape = (block_0, 1024)
        elif x.size % 128 == 0:
            x_2d = x.reshape(-1, 128)
            if x_2d.shape[0] % 512 == 0:
                block_0 = 512
            elif x_2d.shape[0] % 128 == 0:
                block_0 = 128
            elif x_2d.shape[0] % 8 == 0:
                block_0 = 8
            else:
                return jax.nn.sigmoid(x)
            block_shape = (block_0, 128)
        else:
            # Fallback to vanilla JAX if shape doesn't align with TPU block constraints
            return jax.nn.sigmoid(x)
            
        grid_shape = (x_2d.shape[0] // block_shape[0], x_2d.shape[1] // block_shape[1])
        
        out_2d = pl.pallas_call(
            sigmoid_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
            ),
        )(x_2d)
        
        return out_2d.reshape(x.shape)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
