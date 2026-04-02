acc = jnp.zeros((128, 1024), dtype=jnp.float32)
for k in range(64): # 8192 // 128
    x_block = x_ref[:, k*128 : (k+1)*128]
    w_block = weight_ref[k*128 : (k+1)*128, :]
    acc += jnp.dot(x_block, w_block)
acc += bias_ref[...]
acc = jax.nn.gelu(acc)
o_ref[...] = acc
