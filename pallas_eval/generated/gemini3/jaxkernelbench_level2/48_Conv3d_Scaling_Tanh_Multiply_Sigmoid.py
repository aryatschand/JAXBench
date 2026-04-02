')
        )
        
        # Transpose NDHWC -> CNDHW
        x = jnp.transpose(x, (4, 0, 1, 2, 3))
        
        C, N, D, H, W = x.shape
        M = N * D * H * W
        x = x.reshape(C, M)
        
        pad_C = (8 - C % 8) % 8
        pad_M = (1024 - M % 1024) % 1024
        
        if pad_C > 0 or pad_M > 0:
            x = jnp.pad(x, ((0, pad_C), (0, pad_M)))
            
        C_padded, M_padded = x.shape
        
        bc = self.bias_conv.reshape(-1, 1)
        sf = self.scaling_factor.reshape(-1, 1)
        b = self.bias.reshape(-1, 1)
        
        if pad_C > 0:
            bc = jnp.pad(bc, ((0, pad_C), (0, 0)))
            sf = jnp.pad(sf, ((0, pad_C), (0, 0)))
            b = jnp.pad(b, ((0, pad_C), (0, 0)))
            
        bc = jnp.repeat(bc, 128, axis=1)
        sf = jnp.repeat(sf, 128, axis=1)
        b = jnp.repeat(b, 128, axis=1)
        
        grid_shape = (C_padded // 8, M_padded // 1024)
        
        x = pl.pallas_call(
            elementwise_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((8, 1024), lambda i, j: (i, j)),
                    pl.BlockSpec((8, 128), lambda i, j: (i, 0)),
                    pl.BlockSpec((8, 128), lambda i, j: (i, 0)),
                    pl.BlockSpec((8, 128), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((8, 1024), lambda i, j: (i, j)),
            ),
        )(x, bc, sf, b)
        
        if pad_C > 0 or pad_M >
