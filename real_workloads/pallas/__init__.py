"""
Pallas Kernel Evaluation Framework for Real Workloads.

This module provides infrastructure to evaluate Pallas TPU kernels
against JAX baseline implementations from real_workloads/models/.

Similar to KernelBench, it checks:
1. Correctness: Pallas output matches JAX baseline
2. Performance: Speedup over JAX baseline
"""
