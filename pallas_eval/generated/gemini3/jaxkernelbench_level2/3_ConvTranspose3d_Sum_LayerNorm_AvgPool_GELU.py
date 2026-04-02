```python
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def post_conv_kernel(x_ref, bias_ref, sum_weight_ref, norm_weight_ref, norm_bias_ref, o_ref):
    x_blk = x_ref[...]
    bias = bias_ref[...]
    sum_weight = sum_weight_ref[...]
    norm_weight = norm_weight_ref[...]
    norm_bias = norm_bias_ref[...]
    
    # Reshape from 2D back to 1D/scalar for broadcasting
    bias = bias.reshape((64,))
    sum_weight = sum_weight.reshape(())
    norm_weight = norm_weight.reshape((64,))
    norm_bias = norm_bias.reshape((64,))
    
    # Add bias and sum_weight
    x_blk = x_blk + bias + sum_weight
    
    # LayerNorm over channels (axis=-1)
    mean = jnp.mean(x_blk, axis=-1, keepdims=True)
    var = jnp.var(x_blk, axis=-1, keepdims=True)
    x_ln = (x_blk - mean) / jnp.sqrt(var + 1e-5)
    x_ln = x_ln * norm_weight + norm_bias
    
    # AvgPool3d (kernel 2x2x2, stride 2x2x2)
    # x_ln shape: (1, 4, 8, 8, 64)
    # Reshape to split spatial dims into (output_dim, pool_dim)
    x_reshaped = x_ln.reshape((1, 2, 2, 4, 2, 4, 2, 64))
    # Sum over pool_dims (axis 2, 4, 6) and divide by pool volume (8)
    pooled = jnp.sum(x_reshaped, axis=(2, 4, 6)) / 8.0
    
    # Transpose to NCDHW: (1, 2, 4, 4, 64) -> (1, 64, 2, 4, 4)
    out = jnp.transpose(pooled, (0, 4, 1, 2, 3))
    
    # GELU activation
    out = jax.nn.gelu(out)
    
    o_ref[...] = out

class Model:
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_
