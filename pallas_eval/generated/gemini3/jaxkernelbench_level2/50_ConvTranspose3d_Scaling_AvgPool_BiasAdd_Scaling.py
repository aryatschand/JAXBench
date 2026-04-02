h, w: (n, d, h, w, 0)`.
Wait, the grid indices are `n, d, h, w`.
For input, the block is `(1, 2, 2, 2, 16)`.
If grid index is `(n, d, h, w)`, the starting index in the tensor is `(n * 1, d * 2, h * 2, w * 2, 0 * 16)`.
Pallas `BlockSpec` index map specifies the *grid index* multiplier.
Wait, the index map returns the grid indices for each dimension.
For input: `lambda n, d, h, w: (n, d, h, w, 0)`.
Because the block shape is `(1, 2, 2, 2, 16)`, Pallas will automatically multiply the index map result by the block shape!
So `(n, d, h, w, 0)` * `(1, 2, 2, 2, 16)` = `(n, d*2, h*2, w*2, 0)`.
This is EXACTLY what we want!
For output: `lambda n, d, h, w: (n, d, h, w, 0)`.
Block shape is `(1, 1, 1, 1, 16)`.
Starting index: `(n, d, h, w, 0)`.
This is EXACTLY what we want!

Wait, is a block size of `(1, 2, 2, 2, 16)` too small?
Total elements: `1 * 2 * 2 * 2 * 16 = 128`.
128 elements is very small for a TPU kernel block. It might be inefficient because of kernel launch overhead or poor vectorization.
Can we increase the block size?
We want to process multiple elements in the grid.
Let's look at the grid: `(128, 15, 31, 31)`.
Can we block over `N`?
`N=128`. We can use a block size of `128` for `N`!
Input block shape: `(128, 2, 2, 2, 16)`.
Output block shape: `(128, 1, 1, 1, 16)`.
Grid: `(1, 15, 31, 31)`.
Elements per input block: `128 * 8 * 16 = 16384`.
This is a MUCH better block size!
Let's check if it divides evenly:
Input: `128 % 128 == 0`. Yes.
Output: `128 % 128 == 0`. Yes.
