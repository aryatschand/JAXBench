x = jnp.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6) # N, D, H, W, C
orig_out = jnp.transpose(x, (0, 4, 1, 2, 3))

x_flat = x.reshape(2, 3*4*5, 6) # N, S, C
flat_out = jnp.transpose(x_flat, (0, 2, 1)) # N, C, S
new_out = flat_out.reshape(2, 6, 3, 4, 5)

print(jnp.allclose(orig_out, new_out)) # True
