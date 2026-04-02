"""
JAXBench Level 1 - Task 94: MSELoss
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.149871
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mse_kernel(pred_ref, target_ref, out_ref):
    # Read the entire block into VMEM
    pred = pred_ref[...]
    target = target_ref[...]
    
    # Compute squared differences
    diff = pred - target
    sq = diff * diff
    
    B0, B1 = sq.shape
    # Reshape to (B0, B1 // 128, 128) to reduce over the inner dimension
    # while keeping the output block size a multiple of 128 to satisfy TPU constraints.
    sq_reshaped = sq.reshape((B0, B1 // 128, 128))
    
    # Sum over the chunks and write to the output reference
    out_ref[...] = jnp.sum(sq_reshaped, axis=1)

class Model:
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        B0, B1 = predictions.shape
        
        # Fallback to vanilla JAX if shapes are not compatible with our Pallas kernel
        # or if the row size is too large to fit in VMEM (16MB limit).
        # 32 * 131072 * 4 bytes = 16MB.
        if B1 % 128 != 0 or B0 % 32 != 0 or B1 > 131072:
            return jnp.mean((predictions - targets) ** 2)
            
        # Use a block size of 32 for the batch dimension.
        # For B1=32768, VMEM usage per input is 32 * 32768 * 4 bytes = 4MB, which is safe.
        block_0 = 32
        block_1 = B1
        
        grid_shape = (B0 // block_0,)
        
        out = pl.pallas_call(
            mse_kernel,
            out_shape=jax.ShapeDtypeStruct((B0, 128), predictions.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_0, block_1), lambda i: (i, 0)),
                    pl.BlockSpec((block_0, block_1), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_0, 128), lambda i: (i, 0)),
            ),
        )(predictions, targets)
        
        # Sum the partially reduced blocks and divide by the total number of elements
        return jnp.sum(out) / predictions.size

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    scale = jax.random.uniform(key1, shape=())
    return [jax.random.uniform(key2, shape=(batch_size, *input_shape))*scale,
            jax.random.uniform(key3, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return []
