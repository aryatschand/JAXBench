"""Optimized Pallas TPU kernels.

These are hand-written Pallas kernels that can be swapped into the models
via the kernel registry. Each kernel matches the signature of its vanilla
counterpart in kernels.py.

Usage:
    from inference_speedup.kernels import swap_kernel
    from inference_speedup.pallas_kernels import pallas_rmsnorm

    swap_kernel('rmsnorm', pallas_rmsnorm)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
try:
    from jax.experimental.pallas import tpu as pltpu
    HAS_TPU_PALLAS = True
except ImportError:
    HAS_TPU_PALLAS = False
from functools import partial


# ---------------------------------------------------------------------------
# Pallas RMSNorm — fused normalization kernel (tiled for TPU Mosaic)
# ---------------------------------------------------------------------------

def _rmsnorm_kernel_tiled(x_ref, w_ref, o_ref, *, eps):
    """Pallas kernel: fused RMSNorm for a block of rows.

    Operates on (block_rows, D) tiles. Fuses: mean(x^2), rsqrt, scale
    in a single kernel to avoid intermediate materializations.

    Weight is passed as (1, D) and broadcast across rows.
    """
    x = x_ref[...].astype(jnp.float32)   # (block_rows, D)
    w = w_ref[0, :]                        # (D,) — take row 0 of (1, D)

    # RMS normalization
    sq = x * x
    mean_sq = jnp.mean(sq, axis=-1, keepdims=True)  # (block_rows, 1)
    rsqrt_val = jax.lax.rsqrt(mean_sq + eps)

    normed = x * rsqrt_val
    o_ref[...] = (normed * w).astype(o_ref.dtype)


def pallas_rmsnorm(x, weight, eps=1e-6):
    """Pallas-optimized RMSNorm.

    Tiles the input along the batch*seq dimension with proper 2D BlockSpec
    for TPU Mosaic compatibility. Weight is reshaped to 2D to avoid
    Mosaic tiling issues with 1D tensors.
    """
    orig_shape = x.shape
    if x.ndim == 3:
        B, S, D = x.shape
        x_2d = x.reshape(B * S, D)
    else:
        x_2d = x
        D = x.shape[-1]

    BS = x_2d.shape[0]
    # Block size: process 8 rows at a time (TPU rows must be divisible by 8)
    block_rows = min(8, BS)
    n_blocks = BS // block_rows

    # Handle remainder if BS not divisible by block_rows
    if BS % block_rows != 0:
        # Pad to make divisible
        pad_rows = block_rows - (BS % block_rows)
        x_2d = jnp.pad(x_2d, ((0, pad_rows), (0, 0)))
        n_blocks = x_2d.shape[0] // block_rows

    # Reshape weight to 2D for Mosaic compatibility
    w_2d = weight[None, :]  # (1, D)

    out_2d = pl.pallas_call(
        partial(_rmsnorm_kernel_tiled, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
        in_specs=[
            pl.BlockSpec((block_rows, D), lambda i: (i, 0)),
            pl.BlockSpec((1, D), lambda i: (0, 0)),  # broadcast weight
        ],
        out_specs=pl.BlockSpec((block_rows, D), lambda i: (i, 0)),
        grid=(n_blocks,),
    )(x_2d, w_2d)

    # Remove padding and reshape back
    out_2d = out_2d[:BS]
    return out_2d.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Registry of available Pallas kernels
# ---------------------------------------------------------------------------

AVAILABLE_PALLAS_KERNELS = {
    'rmsnorm': pallas_rmsnorm,
}
