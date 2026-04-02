valid_H = H_out * self.kernel_size
                valid_W = W_out * self.kernel_size
                x_valid = x_val[:valid_H, :valid_W]
                x_reshaped = x_valid.reshape((H_out, self.kernel_size, W_out, self.kernel_size))
                out_val = jnp.sum(x_reshaped, axis=(1, 3))
