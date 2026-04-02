for a in axes if a != R and a != S]
        forward_perm = batch_dims + [R, S]
        
        out_axes = [a for a in axes if a != R]
        pallas_out_axes = batch_dims + [S]
        backward_perm = [pallas_out_axes.index(a) for a in out_axes]
        
        # Permute x to (B1, B2, B3, R, S)
        x_perm = jnp.transpose(x, forward_perm)
        shape_perm = x_perm.shape
        
        B = shape_perm[0] * shape_perm[1] * shape_perm[2]
        R_dim = shape_perm[3]
        S_dim = shape_perm[4]
        
        x_flat = x_perm.reshape(B, R_dim, S_dim)
        
        # Pad dimensions
        B_padded = (B + 7) // 8 * 8
        R_padded = (R_dim + 7) // 8 * 8
        S_padded = (S_dim + 127) // 128 * 128
        
        pad_B = B_padded - B
        pad_R = R_padded - R_dim
        pad_S = S_padded - S_dim
        
        # Pad R with inf
        x_padded = jnp.pad(x_flat, ((0, 0), (0, pad_R), (0, 0)), constant_values=jnp.inf)
        # Pad B and S with -inf
        x_padded = jnp.pad(x_padded, ((0, pad_B), (0, 0), (0, pad_S)), constant_values=-jnp.inf)
        
        # Reshape for Pallas
        x_reshaped = x_padded.reshape(B_padded // 8, 8, R_padded, S_padded)
        
        # Pallas call
        grid_shape = (B_padded // 8,)
        out_reshaped = pl.pallas_call(
            min_softmax_kernel,
            out_shape=jax.ShapeDtypeStruct((B_padded // 8, 8, S_padded), x_reshaped.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((1, 8, R_padded, S_padded), lambda i: (i, 0, 0, 0))
                ],
                out_specs=pl.BlockSpec((1, 8, S_padded), lambda i: (i, 0,
