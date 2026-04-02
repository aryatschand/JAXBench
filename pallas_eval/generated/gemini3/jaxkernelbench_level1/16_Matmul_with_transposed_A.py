def matmul_kernel(a_ref, b_ref, c_ref):
    acc = jnp.zeros((128, 256), dtype=jnp.float32)
    def body(k, acc):
        a_tile = a_ref[k, :, :] # (256, 128)
        b_tile = b_ref[k, :, :] # (256, 256)
        return acc + jnp.dot(a_tile.T, b_tile, preferred_element_type=jnp.float32)
    acc = jax.lax.fori_loop(0, 32, body, acc)
    c_ref[...] = acc.astype(c_ref.dtype)
