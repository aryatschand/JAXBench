```python
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def pallas_ln_gelu_scale(x, weight, bias, eps, scaling_factor):
    orig_shape = x.shape
    W = orig_shape[-1]
    B = orig_shape[0] * orig_shape[1] * orig_shape[2] * orig_shape[3]
    
    x_flat = x.reshape((B, W))
    weight_2d = weight.reshape((1, W))
    bias_2d = bias.reshape((1, W))
    
    block_B = 128
    pad_B = (block_B - (B % block_B)) % block_B
    if pad_B > 0:
        x_flat = jnp.pad(x_flat, ((0, pad_B), (0, 0)))
        
    block_W = 128 if W < 128 else (W + 127) // 128 * 128
    pad_W = block_W - W
    if pad_W > 0:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, pad_W)))
        weight_2d = jnp.pad(weight_2d, ((0, 0), (0, pad_W)))
        bias_2d = jnp.pad(bias_2d, ((0, 0), (0, pad_W)))
        
    grid = (x_flat.shape[0] // block_B,)
    
    def ln_kernel(x_ref, w_ref, b_ref, o_ref):
        x_blk = x_ref[:, :]
        w_blk = w_ref[0, :]
        b_blk = b_ref[0, :]
        
        if pad_W > 0:
            x_valid = x_blk[:, :-pad_W]
            w_valid = w_blk[:-pad_W]
            b_valid = b_blk[:-pad_W]
        else:
            x_valid = x_blk
            w_valid = w_blk
            b_valid = b_blk
            
        mean = jnp.mean(x_valid, axis=-1, keepdims=True)
        var = jnp.mean((x_valid - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x_valid - mean) / jnp.sqrt(var + eps)
        x_norm = x_norm * w_valid + b_valid
        
        x_gelu = j
