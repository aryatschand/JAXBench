def fused_kernel(x_ref, total_bias_ref, scale_ref, gn_weight_ref, gn_bias_ref, h_valid_ref, w_valid_ref, out_ref):
    C_G = x_ref.shape[2]
    H_pad = x_ref.shape[3]
    W_pad = x_ref.shape[4]
    
    x = x_ref[0, 0, :, :, :]
    total_bias = total_bias_ref[0, :].reshape(C_G, 1, 1)
    scale = scale_ref[0, :].reshape(C_G, 1, 1)
    gn_weight = gn_weight_ref[0, :].reshape(C_G, 1, 1)
    gn_bias = gn_bias_ref[0, :].reshape(C_G, 1, 1)
    
    h_valid = h_valid_ref[...]
    w_valid = w_valid_ref[...]
    
    x = (x + total_bias) * scale
    x = jax.nn.sigmoid(x)
    
    h_idx = jnp.arange(H_pad).reshape(1, H_pad, 1)
    w_idx = jnp.arange(W_pad).reshape(1, 1, W_pad)
    mask = (h_idx < h_valid) & (w_idx < w_valid)
    
    valid_count = C_G * h_valid * w_valid
    valid_count = valid_count.astype(jnp.float32)
    
    x_masked = jnp.where(mask, x, 0.0)
    mean = jnp.sum(x_masked) / valid_count
    
    diff = jnp.where(mask, x - mean, 0.0)
    var = jnp.sum(diff * diff) / valid_count
    
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    
    out = x_norm * gn_weight + gn_bias
    out_ref[0, 0, :, :, :] = out
