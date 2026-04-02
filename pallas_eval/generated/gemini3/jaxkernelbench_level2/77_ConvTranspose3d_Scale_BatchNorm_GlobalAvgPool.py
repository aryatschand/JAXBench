padded, block_C)
    x_val = x_ref[0, :, :]
    
    # Sum over S_padded
    x_sum = jnp.sum(x_val, axis=0)
    
    # Mean
    x_mean = x_sum / S_orig
    
    # Linear ops
    bias = bias_ref[...]
    mean = mean_ref[...]
    var = var_ref[...]
    bn_weight = bn_weight_ref[...]
    bn_bias = bn_bias_ref[...]
    
    val = x_mean + bias
    val = val * scale_factor
    val = (val - mean) / jnp.sqrt(var + eps)
    val = val * bn_weight + bn_bias
    
    out_ref[0, ...] = val

def pallas_fused_reduce_1d(x_flat, bias, mean, var, bn_weight, bn_bias, scale_factor, eps, S_orig):
    N, S_padded, C = x_flat.shape
    block_C = 128 if C % 128 == 0 else (64 if C % 64 == 0 else (32 if C % 32 == 0 else C))
    block_N = 1
    
    grid = (N // block_N, C // block_C)
    
    return pl.pallas_call(
        fused_reduce_1d_kernel,
        out_shape=jax.ShapeDtypeStruct((N, C), x_flat.dtype),
        scale_factor=scale_factor,
        eps=eps,
        S_orig=S_orig,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_N, S_padded, block_C), lambda n, c: (n, 0, c)),
                pl.BlockSpec((block_C,), lambda n, c: (c,)),
                pl.BlockSpec((block_C,), lambda n, c: (c,)),
                pl.BlockSpec((block_C,), lambda n, c: (c,)),
                pl.BlockSpec((block_C,), lambda n, c: (c,)),
                pl.BlockSpec((block_C,), lambda n, c: (c,)),
            ],
            out_specs=pl.BlockSpec((block_N, block_C), lambda n, c: (n, c)),
        ),
    )(x_flat, bias, mean, var, bn_weight, bn_bias)
```
Wait, `S_padded` is passed as a dimension in `BlockSpec`.
Is `S_padded` static?
In JAX, when `
