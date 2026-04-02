import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def softplus_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.logaddexp(x, 0.0)

class Model:
    """
    Simple model that performs a Softplus activation.
    """
    def __init__(self):
        pass

    def forward(self, x):
        """
        Applies Softplus activation to the input tensor.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with Softplus applied, same shape as input.
        """
        orig_shape = x.shape
        if x.ndim == 1:
            x_2d = x.reshape(1, -1)
        elif x.ndim == 0:
            x_2d = x.reshape(1, 1)
        else:
            x_2d = x.reshape(x.shape[0], -1)
            
        N, M = x_2d.shape
        
        def power_of_two(val):
            if val <= 0: return 1
            return 2**(val.bit_length() - 1)
            
        block_n = min(128, power_of_two(N))
        block_m = min(1024, power_of_two(M))
        
        # For TPU f32, block dims should ideally be multiples of (8, 128).
        block_n = max(8, block_n)
        block_m = max(128, block_m)
        
        pad_n = (block_n - (N % block_n)) % block_n
        pad_m = (block_m - (M % block_m)) % block_m
        
        if pad_n > 0 or pad_m > 0:
            x_2d = jnp.pad(x_2d, ((0, pad_n), (0, pad_m)))
            
        grid_shape = (x_2d.shape[0] // block_n, x_2d.shape[1] // block_m)
        
        out_2d = pl.pallas_call(
            softplus_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_n, block_m), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_n, block_m), lambda i, j: (i, j)),
            ),
        )(x_2d)
        
        if pad_n > 0 or pad_m > 0:
            out_2d = out_2d[:N, :M]
            
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
