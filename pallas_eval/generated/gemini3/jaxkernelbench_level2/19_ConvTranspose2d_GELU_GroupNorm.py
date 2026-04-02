def get_kernel(L, C_G, eps):
    def kernel_fn(x_ref, bias_ref, gn_weight_ref, gn_bias_ref, o_ref):
        x = x_ref[0, 0, :, :]       # (C_G, pad_L)
        bias = bias_ref[0, :, :]    # (C_G, 1)
        gn_w = gn_weight_ref[0, :, :] # (C_G, 1)
        gn_b = gn_bias_ref[0, :, :]   # (C_G, 1)
        
        x = x + bias
        x = jax.nn.gelu(x)
        
        mask = jnp.arange(x.shape[1]) < L
        valid_count = L * C_G
        
        x_masked = jnp.where(mask[None, :], x, 0.0)
        mean = jnp.sum(x_masked) / valid_count
        
        var = jnp.sum(jnp.where(mask[None, :], jnp.square(x - mean), 0.0)) / valid_count
        
        x_norm = (x - mean) / jnp.sqrt(var + eps)
        out = x_norm * gn_w + gn_b
        
        o_ref[0, 0, :, :] = out
    return kernel_fn
