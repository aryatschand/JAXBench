"""Sparse Pallas kernels.

Key findings from sparsity experiments:
- Dynamic indexing (fori_loop + dynamic_slice) causes 27-332x SLOWDOWNS
- Static patterns (low-rank, channel pruning) can achieve speedups
- Best result: Low-rank factorization at 2.43x speedup
"""

SPARSITY_PATTERNS = [
    "low_rank",           # W = U @ V factorization (2.43x speedup)
    "channel_pruned",     # Smaller dense matrices (1.40x speedup)
    "diagonal_block",     # Block-diagonal (1.17x speedup)
    "strided",            # Every Nth column
    "band",               # Band matrix
]
