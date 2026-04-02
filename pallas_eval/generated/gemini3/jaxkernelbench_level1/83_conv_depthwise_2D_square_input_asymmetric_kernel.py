def get_kernel(stride, dilation, K, H_out_pad, W_out_pad):
    def conv_kernel(x_ref, w_ref, b_ref, o_ref):
        x = x_ref[0, 0, :, :]
        w = w_ref[0, :]
        b = b_ref[0]
        
        acc = jnp.full((H_out_pad, W_out_pad), b, dtype=jnp.float32)
            
        for k in range(K):
            x_slice = x[k * dilation : k * dilation + H_out_pad * stride : stride,
                        0 : W_out_pad * stride : stride]
            acc += x_slice * w[k]
            
        o_ref[0, 0, :, :] = acc.astype(o_ref.dtype)
    return conv_kernel
