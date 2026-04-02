import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def tanh_kernel(x_ref, o_ref):
    o_ref[...] = jnp.tanh(x_ref[...])

class Model:
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Tanh activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with Tanh applied, same shape as input.
        """
        orig_shape = x.shape
        
        # Use Pallas kernel for the specific benchmark shape or compatible 2D shapes
        if len(orig_shape) == 2 and orig_shape[0] % 128 == 0 and orig_shape[1] % 1024 == 0:
            block_shape = (128, 1024)
            grid_shape = (orig_shape[0] // block_shape[0], orig_shape[1] // block_shape[1])
            return pl.pallas_call(
                tanh_kernel,
                out_shape=jax.ShapeDtypeStruct(orig_shape, x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid_shape,
                    in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
                    out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
                ),
            )(x)
        else:
            # Fallback to vanilla JAX for other shapes
            return jnp.tanh(x)

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
