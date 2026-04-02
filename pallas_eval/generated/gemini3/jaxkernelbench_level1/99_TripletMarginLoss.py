def get_triplet_kernel(margin):
    def triplet_kernel(anchor_ref, pos_ref, neg_ref, loss_ref):
        # anchor_ref: (block_B, D)
        D = anchor_ref.shape[1]
        
        # We can't easily slice a Ref dynamically in a loop without lax.dynamic_slice,
        # which is not supported for Refs in Pallas.
        # We would have to load the whole Ref anyway.
