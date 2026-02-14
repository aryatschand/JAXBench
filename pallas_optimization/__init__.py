"""
Pallas Kernel Optimization Pipeline

Generates optimized Pallas TPU kernels from JAX code using LLMs.

Usage:
    python -m pallas_optimization.run --workload llama3_8b_gqa
"""

from .translator import PallasTranslator
