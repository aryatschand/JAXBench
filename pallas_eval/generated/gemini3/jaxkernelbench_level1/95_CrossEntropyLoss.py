import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import log_softmax

def cross_entropy_kernel(preds_ref, targets_ref, out_ref):
    # Load blocks into VMEM
    preds = preds_ref[...]
    targets = targets_ref[...]
    
    # Compute max for numerical stability
    m = jnp.max(preds, axis=1, keepdims=True)
    
    # Compute exp and sum
    exp_preds = jnp.exp(preds - m)
    sum_exp = jnp.sum(exp_preds, axis=1, keepdims=True)
    
    # Compute logsumexp
    lse = m + jnp.log(sum_exp)
    
    # Gather target values using a mask to avoid gather indirection overhead
    num_classes = preds_ref.shape[1]
    mask = targets == jnp.arange(num_classes)
    target_vals = jnp.sum(jnp.where(mask, preds, 0.0), axis=1, keepdims=True)
    
    # Compute per-example loss
    loss = lse - target_vals
    
    # Store result
    out_ref[...] = loss

class Model:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        batch_size, num_classes = predictions.shape
        
        # Reshape targets to 2D as required by Pallas
        targets_2d = targets.reshape((batch_size, 1))
        
        # Choose block size for batch dimension (128 ensures we stay well within 16MB VMEM)
        block_B = 128
        grid = (batch_size // block_B,)
        
        out_shape = jax.ShapeDtypeStruct((batch_size, 1), predictions.dtype)
        
        loss_2d = pl.pallas_call(
            cross_entropy_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_B, num_classes), lambda i: (i, 0)),
                    pl.BlockSpec((block_B, 1), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_B, 1), lambda i: (i, 0)),
            )
        )(predictions, targets_2d)
        
        # Average negative log likelihood
        return jnp.mean(loss_2d)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    return [
        jax.random.uniform(key1, shape=(batch_size, *input_shape)),
        jax.random.randint(key2, shape=(batch_size,), minval=0, maxval=num_classes)
    ]

def get_init_inputs():
    return []
