Wait, what if `B_block` is 64? Then `(8, 64)` is not a multiple of `(8, 128)`.
But `batch_size = 128`, so `B_block` will be 128.
If `B_block` is 128, `(8, 128)` is perfect.
Inside the kernel:
