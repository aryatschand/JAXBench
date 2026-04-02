import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def selu_kernel(x_ref, o_ref):
    x = x_ref[...]
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    o_ref[...] = scale * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

class Model:
    """
    Simple model that performs a SELU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies SELU activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with SELU applied, same shape as input.
        """
        if x.ndim == 2 and x.shape[0] % 128 == 0 and x.shape[1] % 1024 == 0:
            grid_shape = (x.shape[0] // 128, x.shape[1] // 1024)
            return pl.pallas_call(
                selu_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid_shape,
                    in_specs=[pl.BlockSpec((128, 1024), lambda i, j: (i, j))],
                    out_specs=pl.BlockSpec((128, 1024), lambda i, j: (i, j)),
                ),
            )(x)
        else:
            original_shape = x.shape
            x_flat = x.reshape(-1)
            
            block_size = 1024
            pad_len = (block_size - (x_flat.size % block_size)) % block_size
            if pad_len > 0:
                x_flat = jnp.pad(x_flat, (0, pad_len))
                
            x_2d = x_flat.reshape(-1, block_size)
            
            N = x_2d.shape[0]
            pad_N = (128 - (N % 128)) % 128
            if pad_N > 0:
                x_2d = jnp.pad(x_2d, ((0, pad_N), (0, 0)))
                
            grid_shape = (x_2d.shape[0] // 128, x_2d.shape[1] // block_size)
            
            out_2d = pl.pallas_call(
                selu_kernel,
                out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid_shape,
                    in_specs=[pl.BlockSpec((128, block_size), lambda i, j: (i, j))],
                    out_specs=pl.BlockSpec((128, block_size), lambda i, j: (i, j)),
                ),
            )(x_2d)
            
            out_flat = out_2d.reshape(-1)
            return out_flat[:x.size].reshape(original_shape)

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
