import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mean_kernel(x_ref, o_ref):
    # Read the entire block into VMEM
    x_val = x_ref[...]
    
    # Cast to float32 for accurate and safe reduction, then compute mean
    sum_val = jnp.sum(x_val.astype(jnp.float32), axis=1)
    mean_val = sum_val / x_val.shape[1]
    
    # Write the result back, casting to the original dtype
    o_ref[...] = mean_val.astype(x_ref.dtype)

class Model:
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        self.dim = dim

    def forward(self, x):
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x: Input array of arbitrary shape.

        Returns:
            Output array with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        # Fallback to standard JAX for unsupported dimensions or shapes
        if self.dim != 1 or len(x.shape) != 3:
            return jnp.mean(x, axis=self.dim)
        
        B, M, N = x.shape
        
        # Check if M is a power of 2 and within VMEM limits
        # 1 * M * 512 * 4 bytes <= 12 MB (safe limit to avoid OOM)
        # M * 512 <= 3,145,728
        if (M & (M - 1) != 0) or (M * 512 > 3145728):
            return jnp.mean(x, axis=self.dim)
            
        block_N = 512
        pad_N = (block_N - (N % block_N)) % block_N
        
        # Pad N dimension to be a multiple of block_N
        # Padding with zeros does not affect the sum, and we divide by M (not N)
        if pad_N > 0:
            x_padded = jnp.pad(x, ((0, 0), (0, 0), (0, pad_N)))
        else:
            x_padded = x
            
        N_padded = x_padded.shape[2]
        grid_shape = (B, N_padded // block_N)
        
        out_padded = pl.pallas_call(
            mean_kernel,
            out_shape=jax.ShapeDtypeStruct((B, N_padded), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((1, M, block_N), lambda i, j: (i, 0, j))
                ],
                out_specs=pl.BlockSpec((1, block_N), lambda i, j: (i, j))
            )
        )(x_padded)
        
        # Slice off the padded columns
        if pad_N > 0:
            return out_padded[:, :N]
        return out_padded

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

batch_size = 128
dim1 = 4096 
dim2 = 4095

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1]
