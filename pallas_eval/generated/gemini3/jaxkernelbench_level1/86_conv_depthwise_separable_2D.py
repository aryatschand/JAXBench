grid_shape = (x_flat.shape[0] // M_block,)
        
        out_flat = pl.pallas_call(
            pointwise_fused_kernel,
            out_shape=jax.ShapeDtypeStruct((x_flat.shape[0], C_out_pad), x_flat.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((M_block, C_pad), lambda i: (i, 0)),
                    pl.BlockSpec((C_pad,), lambda i: (0,)),
                    pl.BlockSpec((C_pad, C_out_pad), lambda i: (0, 0)),
                    pl.BlockSpec((C_out_pad,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((M_block, C_out_pad), lambda i: (i, 0)),
            ),
        )(x_flat, dw_bias, pw_weight, pw_bias)
