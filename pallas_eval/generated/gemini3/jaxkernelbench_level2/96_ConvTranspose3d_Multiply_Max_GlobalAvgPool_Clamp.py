x_max = jnp.max(x_flat, axis=2) # Materializes (2048, 14415) in HBM
x_sum = jnp.sum(x_max, axis=1)
