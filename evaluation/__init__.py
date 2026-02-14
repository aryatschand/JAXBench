"""
Evaluation Framework

Provides correctness validation and performance benchmarking for:
- JAX translations (vs PyTorch baseline)
- Pallas kernels (vs JAX baseline)

Key API:
    evaluate_kernel() - Single kernel evaluation for optimization loops
"""

from .validator import validate_correctness
from .workload_registry import get_workload, list_workloads, WorkloadConfig
