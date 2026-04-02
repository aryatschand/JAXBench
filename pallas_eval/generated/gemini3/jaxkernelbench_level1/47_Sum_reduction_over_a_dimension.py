import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def sum_kernel(x_ref, o_ref):
    x_val = x_ref[...].astype(jnp.float32)
    o_ref[...] = jnp.sum(x_val, axis=1, keepdims=True).astype(x_ref.dtype)

class Model:
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        self.dim = dim

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (jnp.ndarray): Input tensor of shape (..., dim, ...).

        Returns:
            jnp.ndarray: Output tensor after sum reduction, shape (..., 1, ...).
        """
        dim = self.dim
        if dim < 0:
            dim += x.ndim
            
        pre_shape = x.shape[:dim]
        reduce_dim = x.shape[dim]
        post_shape = x.shape[dim+1:]
        
        if reduce_dim == 0:
            return jnp.sum(x, axis=self.dim, keepdims=True)
            
        pre_dim = 1
        for d in pre_shape: 
            pre_dim *= d
            
        post_dim = 1
        for d in post_shape: 
            post_dim *= d
        
        if post_dim > pre_dim:
            block_post = 128
            while block_post < 256 and post_dim > block_post:
                block_post *= 2
            block_pre = 1
        else:
            block_pre = 128
            while block_pre < 256 and pre_dim > block_pre:
                block_pre *= 2
            block_post = 1
            
        # Fallback if VMEM requirement is too high (> 8MB)
        if block_pre * reduce_dim * block_post * 4 > 8 * 1024 * 1024:
            if block_post == 256: block_post = 128
            if block_pre == 256: block_pre = 128
            
        if block_pre * reduce_dim * block_post * 4 > 8 * 1024 * 1024:
            return jnp.sum(x, axis=self.dim, keepdims=True)
            
        pad_pre = (block_pre - (pre_dim % block_pre)) % block_pre
        pad_post = (block_post - (post_dim % block_post)) % block_post
        
        x_reshaped = x.reshape((pre_dim, reduce_dim, post_dim))
        
        if pad_pre > 0 or pad_post > 0:
            x_padded = jnp.pad(x_reshaped, ((0, pad_pre), (0, 0), (0, pad_post)))
        else:
            x_padded = x_reshaped
            
        padded_pre_dim = pre_dim + pad_pre
        padded_post_dim = post_dim + pad_post
        
        grid = (padded_pre_dim // block_pre, padded_post_dim // block_post)
        
        out_padded = pl.pallas_call(
            sum_kernel,
            out_shape=jax.ShapeDtypeStruct((padded_pre_dim, 1, padded_post_dim), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_pre, reduce_dim, block_post), lambda i, j: (i, 0, j))],
                out_specs=pl.BlockSpec((block_pre, 1, block_post), lambda i, j: (i, 0, j)),
            )
        )(x_padded)
        
        out_reshaped = out_padded[:pre_dim, :, :post_dim]
        
        out_shape = list(x.shape)
        out_shape[dim] = 1
        return out_reshaped.reshape(out_shape)

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass

batch_size = 128
dim1 = 4096 
dim2 = 4095
reduce_dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [reduce_dim]
