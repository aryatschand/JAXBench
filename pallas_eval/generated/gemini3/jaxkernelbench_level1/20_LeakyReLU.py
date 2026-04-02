"""
JAXBench Level 1 - Task 20: LeakyReLU
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.128042
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_leaky_relu_kernel(negative_slope):
    def leaky_relu_kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.where(x > 0, x, x * negative_slope)
    return leaky_relu_kernel

class Model:
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        self.negative_slope = negative_slope
    
    def forward(self, x):
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x: Input array of any shape.

        Returns:
            Output array with LeakyReLU applied, same shape as input.
        """
        orig_shape = x.shape
        
        # Flatten to 2D to satisfy Pallas constraints
        if len(orig_shape) == 0:
            x_2d = x.reshape(1, 1)
        elif len(orig_shape) == 1:
            x_2d = x.reshape(1, orig_shape[0])
        else:
            x_2d = x.reshape(orig_shape[0], -1)
            
        b0, b1 = 128, 1024
        
        # Pad to ensure block shape evenly divides the tensor shape
        pad_rows = (b0 - (x_2d.shape[0] % b0)) % b0
        pad_cols = (b1 - (x_2d.shape[1] % b1)) % b1
        
        if pad_rows > 0 or pad_cols > 0:
            x_2d_padded = jnp.pad(x_2d, ((0, pad_rows), (0, pad_cols)))
        else:
            x_2d_padded = x_2d
            
        grid_shape = (x_2d_padded.shape[0] // b0, x_2d_padded.shape[1] // b1)
        
        kernel = get_leaky_relu_kernel(self.negative_slope)
        
        out_2d_padded = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d_padded.shape, x_2d_padded.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec((b0, b1), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((b0, b1), lambda i, j: (i, j)),
            )
        )(x_2d_padded)
        
        # Slice back to original 2D shape if padding was applied
        if pad_rows > 0 or pad_cols > 0:
            out_2d = out_2d_padded[:x_2d.shape[0], :x_2d.shape[1]]
        else:
            out_2d = out_2d_padded
            
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
