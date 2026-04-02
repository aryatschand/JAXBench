def conv_1d_kernel(x_ref, w_ref, bias_ref, y_ref, *, K, D, C_in, has_bias, block_W_out):
    # x_ref: (1, C_in, block_W_in)
    # w_ref: (C_in, block_C_out, K)
    # bias_ref: (block_C_out,)
    # y_ref: (1, block_C_out, block_W_out)
    
    x_block = x_ref[0, :, :] # (C_in, block_W_in)
    w_block = w_ref[...]     # (C_in, block_C_out, K)
    
    acc = jnp.zeros((w_block.shape[1], block_W_out), dtype=jnp.float32)
    
    # Loop over K
    # Since K is small, we can use a Python loop or jax.lax.fori_loop.
    # A Python loop is unrolled and usually faster for K=5.
    for k in range(K):
        start_rel = k * D
        x_slice = jax.lax.dynamic_slice(x_block, (0, start_rel), (C_in, block_W_out))
        w_slice = w_block[:, :, k] # (C_in, block_C_out)
        
        # acc += w_slice^T @ x_slice
        # w_slice.T is (block_C_out, C_in)
        # x_slice is (C_in, block_W_out)
        acc += jnp.dot(w_slice.T, x_slice)
        
    if has_bias:
        bias_val = bias_ref[:] # (block_C_out,)
        acc += bias_val[:, None]
        
    y_ref[0, :, :] = acc
