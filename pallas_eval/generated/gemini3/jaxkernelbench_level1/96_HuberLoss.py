"""
JAXBench Level 1 - Task 96: HuberLoss (reduced size to avoid memory issues)
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def huber_loss_sum_kernel(pred_ref, target_ref, loss_sum_ref):
    pred = pred_ref[...]
    target = target_ref[...]
    
    diff = pred - target
    abs_diff = jnp.abs(diff)
    
    one = jnp.array(1.0, dtype=diff.dtype)
    half = jnp.array(0.5, dtype=diff.dtype)
    
    loss = jnp.where(
        abs_diff < one,
        half * (diff * diff),
        abs_diff - half
    )
    
    # Sum the loss over the block and store it in the (1, 1) output reference
    loss_sum_ref[...] = jnp.sum(loss).reshape((1, 1))


class Model:
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        pass
    
    def set_weights(self, weights_dict):
        """No weights for loss function."""
        pass
    
    def forward(self, predictions, targets):
        """
        Compute Smooth L1 (Huber) loss using a Pallas kernel.
        """
        # Ensure inputs have the same shape
        predictions, targets = jnp.broadcast_arrays(predictions, targets)
        orig_size = predictions.size
        
        # Reshape to 2D for the Pallas kernel
        if predictions.ndim == 0:
            predictions = predictions.reshape(1, 1)
            targets = targets.reshape(1, 1)
        elif predictions.ndim == 1:
            predictions = predictions.reshape(predictions.shape[0], 1)
            targets = targets.reshape(targets.shape[0], 1)
        elif predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            
        shape = predictions.shape
        
        # Fixed block sizes that satisfy TPU constraints (multiples of 8 and 128)
        block_0 = 512
        block_1 = 512
        
        # Pad dimensions to be multiples of the block size
        pad_0 = (block_0 - shape[0] % block_0) % block_0
        pad_1 = (block_1 - shape[1] % block_1) % block_1
        
        if pad_0 > 0 or pad_1 > 0:
            predictions = jnp.pad(predictions, ((0, pad_0), (0, pad_1)))
            targets = jnp.pad(targets, ((0, pad_0), (0, pad_1)))
            
        padded_shape = predictions.shape
        grid = (padded_shape[0] // block_0, padded_shape[1] // block_1)
        
        # The kernel outputs the sum of the loss for each block
        loss_sums = pl.pallas_call(
            huber_loss_sum_kernel,
            out_shape=jax.ShapeDtypeStruct(grid, predictions.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_0, block_1), lambda i, j: (i, j)),
                    pl.BlockSpec((block_0, block_1), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            )
        )(predictions, targets)
        
        # Global reduction and mean calculation
        # Padded elements contribute exactly 0 to the sum, so we divide by the original size
        return jnp.sum(loss_sums) / orig_size


# Test code - REDUCED SIZE to avoid memory issues
batch_size = 4096    # Reduced from 32768
input_shape = (4096,)  # Reduced from (32768,)
dim = 1


def get_inputs():
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    scale = jax.random.uniform(key1, ())
    predictions = jax.random.uniform(key2, (batch_size, *input_shape)) * scale
    targets = jax.random.uniform(key3, (batch_size, *input_shape))
    return [predictions, targets]


def get_init_inputs():
    return []
