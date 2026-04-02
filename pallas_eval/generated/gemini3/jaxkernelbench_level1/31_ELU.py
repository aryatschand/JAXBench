import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        self.alpha = alpha
    
    def forward(self, x):
        """
        Applies ELU activation to the input array.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with ELU applied, same shape as input.
        """
        if x.size == 0:
            return x
            
        alpha = jnp.asarray(self.alpha, dtype=x.dtype)
        
        def elu_kernel(x_ref, o_ref):
            x_val = x_ref[...]
            o_ref[...] = jnp.where(x_val > 0, x_val, alpha * (jnp.exp(x_val) - 1))

        if x.ndim == 2 and x.shape[0] % 128 == 0 and x.shape[1] % 1024 == 0:
            block_0, block_1 = 128, 1024
            grid_0 = x.shape[0] // block_0
            grid_1 = x.shape[1] // block_1
            
            return pl.pallas_call(
                elu_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(grid_0, grid_1),
                    in_specs=[pl.BlockSpec((block_0, block_1), lambda i, j: (i, j))],
                    out_specs=pl.BlockSpec((block_0, block_1), lambda i, j: (i, j)),
                )
            )(x)
            
        orig_shape = x.shape
        x_flat = x.flatten()
        
        block_0 = 128
        block_1 = 1024
        block_size = block_0 * block_1
        
        pad_len = (block_size - (x_flat.size % block_size)) % block_size
        if pad_len > 0:
            x_flat = jnp.pad(x_flat, (0, pad_len))
            
        x_2d = x_flat.reshape(-1, block_1)
        
        grid_0 = x_2d.shape[0] // block_0
        grid_1 = 1
        
        out_2d = pl.pallas_call(
            elu_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_0, grid_1),
                in_specs=[pl.BlockSpec((block_0, block_1), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_0, block_1), lambda i, j: (i, j)),
            )
        )(x_2d)
        
        out_flat = out_2d.flatten()
        if pad_len > 0:
            out_flat = out_flat[:-pad_len]
            
        return out_flat.reshape(orig_shape)

    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return [1.0]
