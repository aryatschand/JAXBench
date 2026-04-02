import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def l2_norm_kernel(x_ref, o_ref):
    x = x_ref[...]
    sq_sum = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    norm = jnp.sqrt(sq_sum)
    norm_repeated = pltpu.repeat(norm, x_ref.shape[1], axis=1)
    o_ref[...] = x / norm_repeated

class Model:
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        pass

    def forward(self, x):
        """
        Applies L2 normalization to the input tensor.

        Args:
            x: Input array of shape (*, dim, *).

        Returns:
            Output array with L2 normalization applied, same shape as input.
        """
        batch_size, dim = x.shape
        block_m = min(batch_size, 128)
        block_n = dim
        
        grid = (batch_size // block_m,)
        
        return pl.pallas_call(
            l2_norm_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i: (i, 0))
            )
        )(x)

    def set_weights(self, weights_dict):
        # No weights to set
        pass

batch_size = 4096  # Reduced from 32768 for memory
dim = 8192  # Reduced from 65535

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
