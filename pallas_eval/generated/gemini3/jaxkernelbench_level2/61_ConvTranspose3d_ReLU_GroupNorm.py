mean = jnp.mean(x_reshaped, axis=(1, 2, 3, 5)) # (N, G)
        var = jnp.mean((x_reshaped - mean[:, None, None, None, :, None]) ** 2, axis=(1, 2, 3, 5)) # (N, G)
