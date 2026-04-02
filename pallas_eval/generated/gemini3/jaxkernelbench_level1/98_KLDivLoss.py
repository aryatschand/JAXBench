import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import softmax

def kldiv_kernel(pred_ref, tgt_ref, out_ref):
    pred = pred_ref[...]
    tgt = tgt_ref[...]
    
    # Compute KL divergence elements for the block
    val = tgt * (jnp.log(tgt) - jnp.log(pred))
    
    # Reduce sum over the block and store in the partial sums output
    out_ref[...] = jnp.sum(val, keepdims=True)

class Model:
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        # Ensure inputs are 2D for Pallas
        orig_shape = predictions.shape
        pred_2d = predictions.reshape(-1, orig_shape[-1])
        tgt_2d = targets.reshape(-1, orig_shape[-1])
        
        M, N = pred_2d.shape
        bm = 512
        bn = 512
        
        # Fallback to vanilla JAX if dimensions are not perfectly divisible by block size
        if M % bm != 0 or N % bn != 0:
            log_predictions = jnp.log(predictions)
            return jnp.mean(targets * (jnp.log(targets) - log_predictions))
            
        grid_shape = (M // bm, N // bn)
        
        # Fused element-wise operations and partial reduction
        partial_sums = pl.pallas_call(
            kldiv_kernel,
            out_shape=jax.ShapeDtypeStruct(grid_shape, predictions.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            ),
        )(pred_2d, tgt_2d)
        
        # Final reduction over the partial sums to compute the global mean
        return jnp.sum(partial_sums) / (M * N)

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    scale = jax.random.uniform(key1)
    pred = softmax(jax.random.uniform(key2, (batch_size, *input_shape)) * scale, axis=-1)
    tgt = softmax(jax.random.uniform(key3, (batch_size, *input_shape)), axis=-1)
    return [pred, tgt]

def get_init_inputs():
    return []
