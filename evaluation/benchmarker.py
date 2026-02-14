"""
Performance Benchmarking

Utilities for timing kernel execution on TPU/GPU.
"""

import time
from typing import Callable, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    time_ms: float
    std_ms: float
    num_iters: int
    num_warmup: int


def benchmark_function(
    fn: Callable,
    inputs: tuple,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> BenchmarkResult:
    """
    Benchmark a JIT-compiled function.

    Args:
        fn: Function to benchmark (should be JIT-compiled)
        inputs: Tuple of inputs to pass to fn
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    import jax

    # JIT compile if not already
    jit_fn = jax.jit(fn)

    # Warmup
    for _ in range(num_warmup):
        output = jit_fn(*inputs)
        output.block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        output = jit_fn(*inputs)
        output.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return BenchmarkResult(
        time_ms=avg_time,
        std_ms=std_time,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )


def benchmark_pytorch_xla(
    fn: Callable,
    inputs: tuple,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> BenchmarkResult:
    """
    Benchmark a PyTorch/XLA function.

    Args:
        fn: PyTorch function to benchmark
        inputs: Tuple of inputs to pass to fn
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    import torch_xla.core.xla_model as xm

    # Warmup
    for _ in range(num_warmup):
        output = fn(*inputs)
        xm.mark_step()
        xm.wait_device_ops()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        output = fn(*inputs)
        xm.mark_step()
        xm.wait_device_ops()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return BenchmarkResult(
        time_ms=avg_time,
        std_ms=std_time,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
