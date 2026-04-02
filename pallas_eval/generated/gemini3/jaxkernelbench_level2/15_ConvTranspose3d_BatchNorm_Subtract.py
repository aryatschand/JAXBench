BlockSpec` maps grid indices to block indices.
If tensor is `(num_blocks * 8, 128)`, and block is `(8, 128)`, then `lambda i: (i, 0)` means grid `i` accesses `[i*8 : i*8+8, 0:128]`.
Yes! In Pallas, the `index_map` returns the *block* index, not the element index.
So `lambda i: (i, 0)` means the block starts at `i * block_shape[0]`.
Let's verify this.
If tensor is `(K, 128)`, block is `(8, 128)`.
Grid is `(K // 8,)`.
`index_map = lambda i: (i, 0)`.
For `i=0`, block is `(0, 0)`, which covers `0:8, 0:128`.
For `i=1`, block is `(1, 0)`, which covers `8:16, 0:128`.
This is perfectly valid and strictly follows the `(8, 128)` rule!

Let's refine this:
```python
        # Broadcast in JAX
        a_broadcast = jnp.broadcast_to(a, x.shape)
        sm_broadcast = jnp.broadcast_to(spatial_mean, x.shape)
        
        # Flatten to 1D
        x_flat = x.reshape(-1)
        a_flat = a_broadcast.reshape(-1)
        sm_flat = sm_broadcast.reshape(-1)
        
        total_elements = x_flat.shape[0]
        
        # Pad to multiple of 1024 (8 * 128)
        pad_len = (1024 - (total_elements % 1024)) % 1024
        if pad_len > 0:
            x_padded = jnp.pad(x_flat, (0, pad_len))
            a_padded = jnp.pad(a_flat, (0, pad_len))
            sm_padded = jnp.pad(sm_flat, (0, pad_len))
        else:
            x_padded = x_flat
            a_padded = a_flat
            sm_padded = sm_flat
            
        # Reshape to 2D: (K, 128) where K is a multiple of 8
        K = x_padded.shape[0] // 128
        x_2d = x_padded.reshape(K, 128)
        a_2d = a_padded.reshape(K, 128)
        sm_2d = sm_padded.reshape(K, 12
