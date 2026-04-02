Wait, `out_shape` is `(N, D_out, H_out, W_out, C)`.
The `out_specs` is `BlockSpec((1, 1, 1, W_out, 1), lambda n, c, d, h: (n, d, h, 0, c))`.
Wait, the block shape is `(1, 1, 1, W_out, 1)`.
But the output tensor has shape `(N, D_out, H_out, W_out, C)`.
The index map is `lambda n, c, d, h: (n, d, h, 0, c)`.
This means:
dim 0: `n` (size 1)
dim 1: `d` (size 1)
dim 2: `h` (size 1)
dim 3: `0` (size `W_out`)
dim 4: `c` (size 1)
This perfectly matches the block shape `(1, 1, 1, W_out, 1)`!
Wait, the block shape must match the rank of the tensor.
The tensor is 5D. The block shape is 5D.
The index map returns a 5-tuple.
This is perfectly correct.

Let's double check the `post_conv` grid and specs.
