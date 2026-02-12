"""
Pallas Kernel Optimization Pipeline

This module provides tools for generating, evaluating, and benchmarking
Pallas TPU kernels. It explores whether LLM-generated Pallas kernels can
achieve speedups over JAX's native implementations.

Key findings:
- JAX's XLA compiler is highly optimized for standard operations
- Custom Pallas adds 46-110x overhead for basic matmul
- Pallas is useful for operations JAX doesn't support natively

Usage:
    from pallas.kernels.matmul import matmul_pallas
    from pallas.utils import benchmark_fn, run_on_tpu
"""

__version__ = "0.1.0"
