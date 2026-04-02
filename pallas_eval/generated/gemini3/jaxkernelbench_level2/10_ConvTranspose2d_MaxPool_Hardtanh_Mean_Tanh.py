import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def hardtanh_mean_tanh_kernel(x_ref, out_ref, min_ref, max_ref):
    min_val = min_ref[0]
    max_val = max_ref[0]
    
    # Read the entire block from HBM to VMEM
    x_val = x_ref[...]
    
    # Hardtanh (clip)
    x_val = jnp.clip(x_val, min_val, max_val)
    
    # Mean over the spatial dimensions (axis 1)
    # Using sum and division to ensure safe f32 reduction in Mosaic
    sum_val = jnp.sum(x_val, axis=1, keepdims=True)
    mean_val = sum_val / x_val.shape[1]
    
    # Tanh
    out_val = jnp.tanh(mean_val)
    
    # Write the result back to HBM
    out_ref[...] = out_val

def pallas_hardtanh_mean_tanh(x_nchw, hardtanh_min, hardtanh_max):
    N, C, H, W = x_nchw.shape
    NC = N * C
    HW = H * W
    
    # Reshape to 2D: (N*C, H*W) to process each spatial map independently
    x_reshaped = x_nchw.reshape(NC, HW)
    
    # Choose block_nc to be a multiple of 8, and HW is a multiple of 128 (16384)
    # 64 * 16384 * 4 bytes = 4 MB, which fits perfectly in the 16 MB TPU VMEM.
    block_nc = 64
    while NC % block_nc != 0 and block_nc > 8:
        block_nc //= 2
        
    grid_shape = (NC // block_nc,)
    
    min_arr = jnp.array([hardtanh_min], dtype=x_nchw.dtype)
    max_arr = jnp.array([hardtanh_max], dtype=x_nchw.dtype)
    
    out = pl.pallas_call(
        hardtanh_mean_tanh_kernel,
        out_shape=jax.ShapeDtypeStruct((NC, 1), x_nchw.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((block_nc, HW), lambda i: (i, 0
