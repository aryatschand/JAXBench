and solves ALL the multiple-of-8 and evenly-dividing problems!

Let's verify the block sizes with this new padded shape.
Output shape: `(16, 64, 64, 64, 64)`.
Grid: `(16, 2, 2, 2)`.
Output block shape: `(1, 64, 32, 32, 32)`.
Does `(1, 64, 32, 32, 32)` evenly divide `(16, 64, 64, 64, 64)`? YES!
Input `x_padded` shape: `(16, 8, 66, 68, 70)`.
Input block shape: `(1, 8, 66, 68, 70)`.
Does `(1, 8, 66, 68, 70)` evenly divide `(16, 8, 66, 68, 70)`? YES!
Size of input block: `8 * 66 * 68 * 70 * 4 = 10` MB.
Size of output block: `64 * 32 * 32 * 32 * 4 = 8.3` MB.
Total VMEM: ~18.3 MB.
Wait! 18.3 MB EXCEEDS the 16 MB VMEM limit of TPU v6e!
Ah! We need to be careful.
Can we use a smaller output block?
Grid: `(16, 4, 4, 4)`.
Output block shape: `(1, 64, 16, 16, 16)`.
Does `16` evenly divide `64`? YES!
Input block shape: `(1, 8, 66, 68, 70)`.
Size of input block: 10 MB.
Size of output block: `64 * 16 * 16 * 16 * 4 = 1` MB.
Total VMEM: 11 MB. This FITS in 16 MB!
Let's check the `dot_general` size for this block.
Spatial size: `16 * 16 * 16 = 4096`.
`4096` is a multiple of 128.
Contracting dim: 8.
Non-contracting dims: 4096 and 64.
All are multiples of 8!
This is absolutely perfect.

Let's refine the kernel with Grid `(16, 4, 4, 4)`.
Grid dimensions:
`b_grid = pl.program_id(0)`
`d_grid = pl.
