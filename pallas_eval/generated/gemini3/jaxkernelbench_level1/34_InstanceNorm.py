import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        self.num_features = num_features
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            Output tensor with Instance Normalization applied, same shape as input.
        """
        B, C, H, W = x.shape
        N = H * W
        M = B * C
        
        # Fallback to standard JAX if N is not a multiple of 128 or if it's too large for VMEM
        if N % 128 != 0 or N > 131072:
            mean = jnp.mean(x, axis=(2, 3), keepdims=True)
            var = jnp.var(x, axis=(2, 3), keepdims=True)
            return (x - mean) / jnp.sqrt(var + self.eps)
            
        x_reshaped = x.reshape((M, N))
        
        # Pad M to be a multiple of 8 to satisfy TPU block shape constraints for f32/bf16
        pad_M = (8 - (M % 8)) % 8
        if pad_M > 0:
            x_reshaped = jnp.pad(x_reshaped, ((0, pad_M), (0, 0)))
            
        M_padded = x_reshaped.shape[0]
        block_M = 8
        grid = (M_padded // block_M,)
        
        # Cast constants to input dtype to prevent unwanted type promotion (e.g., to float64)
        eps_val = jnp.array(self.eps, dtype=x.dtype)
        N_val = jnp.array(N, dtype=x.dtype)
        
        def kernel(x_ref, o_ref):
            x_val = x_ref[...]
            
            # Compute mean
            mean = jnp.sum(x_val, axis=1, keepdims=True) / N_val
            mean_bcast = pltpu.repeat(mean, N, axis=1)
            
            # Center the values
            x_centered = x_val - mean_bcast
            
            # Compute variance
            var = jnp.sum(x_centered * x_centered, axis=1, keepdims=True) / N_val
            var_bcast = pltpu.repeat(var, N, axis=1)
            
            # Normalize and write output
            o_ref[...] = x_centered / jnp.sqrt(var_bcast + eps_val)
            
        out_reshaped = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M_padded, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_M, N), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_M, N), lambda i: (i, 0))
            )
        )(x_reshaped)
        
        # Remove padding if it was added
        if pad_M > 0:
            out_reshaped = out_reshaped[:M, :]
            
        return out_reshaped.reshape((B, C, H, W))

batch_size = 16  # Reduced from 112 for memory
features = 64
dim1 = 256  # Reduced from 512
dim2 = 256  # Reduced from 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
