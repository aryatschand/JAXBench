def pool_sum_kernel_5d(x_ref, o_ref):
    x = x_ref[...] # shape (1, 6, 6, 6, C)
    x = x.reshape(-1, x.shape[-1]) # shape (216, C)
    x_max = jnp.max(x, axis=0) # shape (C,)
    out = jnp.sum(x_max) # shape ()
    o_ref[...] = out
