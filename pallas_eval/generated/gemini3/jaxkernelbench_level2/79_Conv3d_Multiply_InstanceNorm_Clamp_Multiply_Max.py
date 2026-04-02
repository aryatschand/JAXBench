0))`
Wait, `xm` is `(N, pad_S, C)`.
Block shape: `(1, 1024, C)`.
Index map: `lambda n, s: (n, s, 0)`.
This perfectly slices `xm[n, s*1024 : (s+1)*1024, :]`!

- `mean`: `BlockSpec((1, C), lambda n, s: (n, 0))`
Wait, `mean` is `(N, C)`.
Block shape: `(1, C)`.
Index map: `lambda n, s: (n, 0)`.
This perfectly loads `mean[n, :]`!

- `var`: `BlockSpec((1, C), lambda n, s: (n, 0))`

- `iw`: `BlockSpec((1, C), lambda n, s: (0, 0))`
- `ib`: `BlockSpec((1, C), lambda n, s: (0, 0))`
- `m`: `BlockSpec((1, C), lambda n, s: (0, 0))`
- `cmin`: `BlockSpec((1, 1), lambda n, s: (0, 0))`
- `cmax`: `BlockSpec((1, 1), lambda n, s: (0, 0))`

- `out`: `BlockSpec((1, 1024), lambda n, s: (n, s))`
Wait, `out` shape is `(N, pad_S)`.
Block shape: `(1, 1024)`.
Index map: `lambda n, s: (n, s)`.

Let's check the kernel function:
```python
def fused_kernel(xm_ref, mean_ref, var_ref, iw_ref, ib_ref, m_ref, cmin_ref, cmax_ref, out_ref):
    xm = xm_ref[...] # (1, 1024, C)
    mean = mean_ref[...] # (1, C)
    var = var_ref[...] # (1, C)
    iw = iw_ref[...] # (1, C)
    ib = ib_ref[...] # (1, C)
    m = m_ref[...] # (1, C)
    cmin = cmin_ref[0, 0]
    cmax = cmax_ref[0, 0]
    
    # To broadcast mean and var over the 1024 dimension, we can just reshape them to (1, 1, C)
    # Actually, xm is (1, 1024, C) and mean is (1, C).
    # In JAX, (1, 1024, C) and (1, C) do NOT broadcast directly because the number of dimensions differ.
