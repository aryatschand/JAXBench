pad_elements = (1024 - total_elements % 1024) % 1024
    
    if pad_elements > 0:
        x_flat = jnp.pad(x.flatten(), (0, pad_elements)).reshape(-1, 128)
    else:
        x_flat = x.reshape(-1, 128)
        
    conv_bias_1d = conv_bias.reshape(C)
    bias_1d = bias.reshape(C)
    
    pad_len = (128 - C % 128) % 128
    cb_padded = jnp.pad(conv_bias_1d, (0, pad_len))
    b_padded = jnp.pad(bias_1d, (0, pad_len))
    
    C_padded = cb_padded.shape[0]
    
    cb_2d = jnp.broadcast_to(cb_padded[:, None], (C_padded, 128))
    b_2d = jnp.broadcast_to(b_padded[:, None], (C_padded, 128))
    
    def act_bias_kernel(x_ref, cb_ref, b_ref, o_ref):
        x_val = x_ref[...]
        cb_local = cb_ref[...]
        b_local = b_ref[...]
        
        row = pl.program_id(0)
        r_idx = row * 8 + jnp.arange(8)[:, None]
        c_idx = jnp.arange(128)[None, :]
        flat_indices = r_idx * 128 + c_idx
        
        channel_indices = flat_indices % C
        
        cb_val = cb_local[channel_indices, 0]
        b_val = b_local[channel_indices, 0]
        
        x_val = x_val + cb_val
        
        x_val = jnp.maximum(x_val, 0.0)
        x_val = jnp.where(x_val >= 0, x_val, 0.01 * x_val)
        x_val = 0.5 * x_val * (1.0 + jax.lax.erf(x_val / jnp.sqrt(2.0)))
        x_val = 1.0 / (1.0 + jnp.exp(-x_val))
        
        x_val = x_val + b_val
        
        o_ref[...] = x_val

    out_flat = pl.pallas_call(
        act_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat
