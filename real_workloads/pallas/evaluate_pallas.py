#!/usr/bin/env python3
"""
Pallas Kernel Evaluation Script for Real Workloads.

Evaluates Pallas TPU kernels against JAX baseline implementations.
Similar to KernelBench, checks for correctness and performance.

Usage:
    # List available workloads
    python evaluate_pallas.py --list

    # Evaluate a specific Pallas kernel file against a workload
    python evaluate_pallas.py --workload llama3_8b_gqa --kernel kernels/llama3_gqa_pallas.py

    # Evaluate all kernels in a directory
    python evaluate_pallas.py --kernel-dir kernels/ --all

    # Just run baseline (no Pallas)
    python evaluate_pallas.py --workload llama3_8b_gqa --baseline-only
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import importlib.util

import jax
import jax.numpy as jnp

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from workload_registry import get_workload, list_workloads, WorkloadConfig


def load_pallas_kernel(kernel_path: str) -> callable:
    """
    Load a Pallas kernel from a Python file.

    The kernel file must define a function called `pallas_kernel`
    that takes the same inputs as the baseline and returns the same output.
    """
    spec = importlib.util.spec_from_file_location("pallas_kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'pallas_kernel'):
        raise ValueError(f"Kernel file {kernel_path} must define 'pallas_kernel' function")

    return module.pallas_kernel


def check_correctness(
    baseline_output: jnp.ndarray,
    pallas_output: jnp.ndarray,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Dict[str, Any]:
    """
    Check if Pallas output matches baseline within tolerance.

    Returns dict with:
    - correct: bool
    - max_abs_diff: float
    - max_rel_diff: float
    - mean_abs_diff: float
    """
    abs_diff = jnp.abs(baseline_output - pallas_output)
    max_abs_diff = float(jnp.max(abs_diff))
    mean_abs_diff = float(jnp.mean(abs_diff))

    # Relative diff (avoid division by zero)
    rel_diff = abs_diff / (jnp.abs(baseline_output) + 1e-8)
    max_rel_diff = float(jnp.max(rel_diff))

    # Check if within tolerance
    correct = bool(jnp.allclose(baseline_output, pallas_output, rtol=rtol, atol=atol))

    return {
        'correct': correct,
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_abs_diff': mean_abs_diff,
        'rtol': rtol,
        'atol': atol,
    }


def benchmark_function(
    fn: callable,
    inputs: tuple,
    num_warmup: int = 5,
    num_iters: int = 50,
) -> Dict[str, float]:
    """
    Benchmark a JIT-compiled function.

    Returns dict with:
    - time_ms: average time in milliseconds
    - std_ms: standard deviation
    """
    # JIT compile
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

    return {
        'time_ms': avg_time,
        'std_ms': std_time,
        'num_iters': num_iters,
    }


def evaluate_workload(
    workload: WorkloadConfig,
    pallas_kernel: Optional[callable] = None,
    num_warmup: int = 5,
    num_iters: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a Pallas kernel against JAX baseline for a workload.

    Returns evaluation results including correctness and performance.
    """
    results = {
        'workload': workload.name,
        'model': workload.model,
        'category': workload.category,
        'config': workload.config,
        'timestamp': datetime.now().isoformat(),
    }

    # Generate inputs
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {workload.name}")
        print(f"Model: {workload.model}, Category: {workload.category}")
        print(f"{'='*60}")

    inputs = workload.input_generator()

    if verbose:
        print(f"\nInput shapes:")
        for i, inp in enumerate(inputs):
            print(f"  [{i}]: {inp.shape} ({inp.dtype})")

    # Run baseline
    if verbose:
        print(f"\nRunning JAX baseline...")

    baseline_output = workload.baseline_fn(*inputs)
    baseline_perf = benchmark_function(workload.baseline_fn, inputs, num_warmup, num_iters)

    results['baseline'] = {
        'output_shape': list(baseline_output.shape),
        'output_dtype': str(baseline_output.dtype),
        **baseline_perf,
    }

    if verbose:
        print(f"  Output shape: {baseline_output.shape}")
        print(f"  Time: {baseline_perf['time_ms']:.3f} ms (+/- {baseline_perf['std_ms']:.3f})")

    # Run Pallas kernel if provided
    if pallas_kernel is not None:
        if verbose:
            print(f"\nRunning Pallas kernel...")

        try:
            pallas_output = pallas_kernel(*inputs)
            pallas_output.block_until_ready()

            # Check correctness
            correctness = check_correctness(
                baseline_output, pallas_output,
                rtol=workload.rtol, atol=workload.atol
            )

            # Benchmark
            pallas_perf = benchmark_function(pallas_kernel, inputs, num_warmup, num_iters)

            # Calculate speedup
            speedup = baseline_perf['time_ms'] / pallas_perf['time_ms']

            results['pallas'] = {
                'output_shape': list(pallas_output.shape),
                'output_dtype': str(pallas_output.dtype),
                **pallas_perf,
                **correctness,
                'speedup': speedup,
            }

            if verbose:
                status = "PASS" if correctness['correct'] else "FAIL"
                print(f"  Output shape: {pallas_output.shape}")
                print(f"  Time: {pallas_perf['time_ms']:.3f} ms (+/- {pallas_perf['std_ms']:.3f})")
                print(f"  Correctness: {status}")
                print(f"    Max abs diff: {correctness['max_abs_diff']:.6f}")
                print(f"    Max rel diff: {correctness['max_rel_diff']:.6f}")
                print(f"  Speedup: {speedup:.2f}x")

        except Exception as e:
            results['pallas'] = {
                'error': str(e),
                'correct': False,
            }
            if verbose:
                print(f"  ERROR: {e}")

    return results


def print_summary(results: list):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Workload':<25} | {'Baseline (ms)':<12} | {'Pallas (ms)':<12} | {'Speedup':<8} | {'Status'}")
    print("-" * 80)

    for r in results:
        workload = r['workload']
        baseline_ms = r['baseline']['time_ms']

        if 'pallas' in r and 'time_ms' in r['pallas']:
            pallas_ms = r['pallas']['time_ms']
            speedup = r['pallas'].get('speedup', 0)
            correct = r['pallas'].get('correct', False)
            status = "PASS" if correct else "FAIL"
            print(f"{workload:<25} | {baseline_ms:>12.3f} | {pallas_ms:>12.3f} | {speedup:>7.2f}x | {status}")
        elif 'pallas' in r and 'error' in r['pallas']:
            print(f"{workload:<25} | {baseline_ms:>12.3f} | {'ERROR':<12} | {'-':<8} | FAIL")
        else:
            print(f"{workload:<25} | {baseline_ms:>12.3f} | {'N/A':<12} | {'-':<8} | BASELINE")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pallas kernels against JAX baselines")
    parser.add_argument('--list', action='store_true', help='List available workloads')
    parser.add_argument('--workload', type=str, help='Workload name to evaluate')
    parser.add_argument('--kernel', type=str, help='Path to Pallas kernel file')
    parser.add_argument('--kernel-dir', type=str, help='Directory containing kernel files')
    parser.add_argument('--all', action='store_true', help='Evaluate all workloads')
    parser.add_argument('--baseline-only', action='store_true', help='Only run baseline (no Pallas)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=50, help='Number of benchmark iterations')

    args = parser.parse_args()

    print("=" * 60)
    print("PALLAS KERNEL EVALUATION FRAMEWORK")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    if args.list:
        print("\nAvailable workloads:")
        for name in list_workloads():
            workload = get_workload(name)
            print(f"  - {name} ({workload.model}/{workload.category})")
        return

    results = []

    if args.all:
        workload_names = list_workloads()
    elif args.workload:
        workload_names = [args.workload]
    else:
        print("\nNo workload specified. Use --list to see available workloads.")
        print("Use --workload <name> or --all to run evaluations.")
        return

    for workload_name in workload_names:
        workload = get_workload(workload_name)

        pallas_kernel = None
        if args.kernel and not args.baseline_only:
            pallas_kernel = load_pallas_kernel(args.kernel)
        elif args.kernel_dir and not args.baseline_only:
            # Look for kernel file matching workload name
            kernel_dir = Path(args.kernel_dir)
            kernel_file = kernel_dir / f"{workload_name}.py"
            if kernel_file.exists():
                pallas_kernel = load_pallas_kernel(str(kernel_file))

        result = evaluate_workload(
            workload,
            pallas_kernel=pallas_kernel,
            num_warmup=args.warmup,
            num_iters=args.iters,
        )
        results.append(result)

    print_summary(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
