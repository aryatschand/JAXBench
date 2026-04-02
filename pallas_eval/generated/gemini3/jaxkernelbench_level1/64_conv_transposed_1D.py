N, C_in, L = x.shape
        K = self.kernel_size
        C_out = self.weight.shape[1]
        
        L_out = L + K - 1
        L_block = 1024
        
        num_blocks = (L_out + L_block - 1) // L_block
        L_out_padded = num_blocks * L_block
        
        L_pad_required = (num_blocks + 1) * L_block
        
        left_pad = K - 1
        right_pad = L_pad_required - L - left_pad
        
        x_pad = jnp.pad(x, ((0, 0), (0, 0), (left_pad, right_pad)))
