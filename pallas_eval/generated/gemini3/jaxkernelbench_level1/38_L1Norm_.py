import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def l1_norm_kernel(x_ref, o_ref):
    x = x_ref[...]
    sum_abs = jnp.sum(jnp.abs(x), axis=1, keepdims=True)
    mean_abs = sum_abs / x.shape[1]
    o_ref[...] = x / mean_abs

class Model:
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        pass

    def forward(self, x):
        """
        Applies L1 normalization to the input tensor.

        Args:
            x: Input array of shape (..., dim, ...).

        Returns:
            Output array with L1 normalization applied, same shape as input.
        """
        # Fallback to standard JAX if the shape is not 2D or not aligned to TPU block requirements
        if len(x.shape) != 2 or x.shape[0] % 8 != 0 or x.shape[1] % 128 != 0:
            return x / jnp.mean(jnp.abs(x), axis=1, keepdims=True)
        
        B, D = x.shape
        
        # Determine a suitable block size for the batch dimension
        # We want to fit the block (block_b, D) comfortably in VMEM (e.g., <= 4MB)
        block_b = 128
        while block_b > 8 and (B % block_b != 0 or block_b * D > 1048576):
            block_b //= 2
            
        if B % block_b != 0:
            return x / jnp.mean(jnp.abs(x), axis=1, keepdims=True)
            
        grid = (B // block_b,)
        
        return pl.pallas_call(
            l1_norm_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_b, D), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_b, D), lambda i: (i, 0))
            )
        )(x)

    def set_weights(self, weights_dict):
        """
        No weights to set for this model.
        """
        pass

batch_size = 4096  # Reduced from 32768 for memory
dim = 8192  # Reduced from 65535

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
