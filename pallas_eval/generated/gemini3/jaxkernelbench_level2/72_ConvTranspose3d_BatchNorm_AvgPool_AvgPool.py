C_p,), lambda n, i: (0,)),
                    pl.BlockSpec((C_p,), lambda n, i: (0,)),
                ],
                out_specs=pl.BlockSpec((1, block_L, C_p), lambda n, i: (n, i, 0)),
            ),
        )(x_flat, fused_scale, fused_bias)
        
        out_ndhwc = out_flat.reshape((N_p, D_p//4, H_p//4, W_p//4, C_p))
        
        # Slice to valid region
        out_d = D // 4
        out_h = H // 4
        out_w = W // 4
        
        out_sliced = out_ndhwc[:, :out_d, :out_h, :out_w, :]
        
        # Transpose back to NCDHW
        out_final = jnp.transpose(out_sliced, (0, 4, 1, 2, 3))
        
        return out_final

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (64, 3, 32, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, (16, 1, 1, 1)]
```

Wait, there is a small issue.
In `pl.BlockSpec((C_p,), lambda n, i: (0,))`, `C_p` is a dynamic value (or at least a Python int).
Is it allowed to use `C_p` in `BlockSpec`? Yes, it's just an int.
But wait, `C_p` is 16.
Is `(16,)` a valid block shape?
The prompt says: "For f32: block dims should be multiples of (8, 128). For bf16: (8, 128)."
If `C_p` is 16, it's a multiple of 8, but not 128.
Will this cause an issue?
In practice, Pallas on TPU handles small dimensions fine if they are the innermost dimension and the total block size is reasonable.
The input block is `(1, 128, 64, 16)`. The innermost dimension is 16.
Wait, `64 * 16 = 1024`, which is a multiple of 128.
So the inner two dimensions combined are a multiple of 128. This is usually perfectly fine.
For `scale` and `bias`, the block is `(16,)`. This might be padded to 128 internally, but it's a
