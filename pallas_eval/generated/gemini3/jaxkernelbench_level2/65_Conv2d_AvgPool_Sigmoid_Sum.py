# wait, P is captured from closure
            bias = bias_ref[...] # (out_channels,)
            
            # Reshape to (P, W_pool_padded, P, out_channels)
            x_block = x_block.reshape((P, W_pool_padded, P, out_channels))
            
            # Transpose to (W_pool_padded, P, P, out_channels)
            x_block = jnp.transpose(x_block, (1, 0, 2, 3))
            
            # Add bias
            x_block = x_block + bias.reshape((1, 1, 1, out_channels))
            
            # Mean over PxP window
            x_pool = jnp.sum(x_block, axis=(1, 2)) / float(P * P) # (W_pool_padded, out_channels)
            
            # Sigmoid
            x_sig = jax.nn.sigmoid(x_pool) # (W_pool_padded, out_channels)
            
            # Sum over channels
            x_sum = jnp.sum(x_sig, axis=1) # (W_pool_padded,)
            
            # Mask out the padded elements
            h_idx = pl.program_id(1)
            valid_h = h_idx < H_pool
            
            w_idx = jnp.arange(W_pool_padded)
            valid_w = w_idx < W_pool
            
            mask = valid_h & valid_w # (W_pool_padded,)
            
            x_sum = jnp.where(mask, x_sum, 0.0)
            
            o_ref[...] = jnp.sum(x_sum).reshape((1, 1))
```
This is extremely elegant and fully general!

Let's trace with `P=4`, `H_conv=382`, `W_conv=382`.
`w_align = lcm(128, 4) = 128`.
`W_padded = ((382 + 127) // 128) * 128 = 509 // 128 * 128 = 3 * 128 = 384`.
`W_pool_padded = 384 // 4 = 96`.
`H_padded = ((382 + 3) // 4) * 4 = 385 // 4 * 4 = 96 * 4 = 384`.
`H_pool_padded = 384 // 4 = 96`.
`x_padded` shape: `(batch_size, 384, 384, out_channels)`.
Grid: `(batch
