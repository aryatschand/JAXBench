b0, 8)
        while x.shape[0] % b0 != 0 and b0 > 1:
            b0 //= 2
            
        block_shape = (b0, x.shape[1])
        grid = (x.shape[0] // b0,)
        
        return pl.pallas_call(
            log_softmax_kernel_axis1,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block_shape, lambda i: (i, 0))],
                out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0)),
            )
        )(x)
    elif axis == 0:
        col_bytes = x.shape[0] * x.dtype.itemsize * 2
        max_b1 = (28 * 1024 * 1024) // col_bytes
        max_b1 = max(1, max_b1)
        b1 = 2 ** int(jnp.log2(max_b1))
        b1 = min(b1, 128)
        while x.shape[1] % b1 != 0 and b1 > 1:
            b1 //= 2
            
        block_shape = (x.shape[0], b1)
        grid = (x.shape[1] // b1,)
        
        return pl.pallas_call(
            log_softmax_kernel_axis0,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block_shape, lambda i: (0, i))],
                out_specs=pl.BlockSpec(block_shape, lambda i: (0, i)),
            )
        )(x)
    else:
        return jax.nn.log_softmax(x, axis=axis)
```
Wait, `b0 = min(b0, 8)` has a typo in my thought process: `b0 = min(b0, 8)`.
Let's fix that.

Is there any issue with `x.dtype.itemsize`?
If `x` is a JAX array, `x.dtype.itemsize` works.
Let's test it mentally: `jnp.ones((2,2), dtype=jnp.float32).dtype.itemsize` is 4. Yes.

Wait, what if `axis` is negative? `axis =
