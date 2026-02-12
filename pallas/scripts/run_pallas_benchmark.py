#!/usr/bin/env python3
"""
Pallas Kernel Benchmark Runner

Main entry point for benchmarking Pallas kernels on TPU.
Supports local execution (on TPU) or remote execution via SSH.

Usage:
    # Run on TPU directly
    PJRT_DEVICE=TPU python pallas/scripts/run_pallas_benchmark.py --size 4096

    # Run specific kernel type
    python pallas/scripts/run_pallas_benchmark.py --kernel int5 --size 8192

    # Run all quantization benchmarks
    python pallas/scripts/run_pallas_benchmark.py --all-quant --sizes "4096,8192"
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def run_local_benchmark(args):
    """Run benchmark locally (on TPU)."""
    import jax
    import jax.numpy as jnp

    print("=" * 80)
    print("PALLAS KERNEL BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Sizes: {args.sizes}")
    print()

    results = []

    for size in args.sizes:
        print(f"\n--- Size {size}x{size} ---")

        # Import benchmark modules
        if args.kernel == "matmul" or args.all_quant:
            from pallas.kernels.matmul.tiled_matmul import benchmark_matmul
            result = benchmark_matmul(size)
            result['kernel'] = 'matmul'
            results.append(result)
            print(f"Matmul: JAX={result['jax_ms']:.2f}ms, Pallas={result['pallas_ms']:.2f}ms, "
                  f"Speedup={result['speedup']:.2f}x")

        if args.kernel == "int5" or args.all_quant:
            # Run INT5 comparison
            from pallas.kernels.quantization.int5_pallas import run_all_benchmarks
            int5_results = run_all_benchmarks(size)
            results.append({'kernel': 'int5', 'size': size, **int5_results})

        if args.kernel == "exotic" or args.all_quant:
            # Run exotic formats
            from pallas.kernels.quantization.exotic_formats import run_all_benchmarks
            exotic_results = run_all_benchmarks((size, size))
            results.append({'kernel': 'exotic', 'size': size, 'results': exotic_results})

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"pallas/results/benchmark_{timestamp}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'sizes': args.sizes,
            'kernel': args.kernel,
            'results': results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    return results


def run_remote_benchmark(args):
    """Run benchmark on remote TPU via SSH."""
    from pallas.utils.tpu_runner import run_on_tpu, TPUConfig, check_tpu_connection

    config = TPUConfig(ip=args.tpu_ip) if args.tpu_ip else None

    print("Checking TPU connection...")
    if not check_tpu_connection(config):
        print("ERROR: Cannot connect to TPU")
        return None

    print("TPU connected. Running benchmark...")

    # Copy this script and run it on TPU
    script_path = os.path.abspath(__file__)
    sizes_str = ",".join(str(s) for s in args.sizes)
    remote_args = f"--kernel {args.kernel} --sizes {sizes_str} --local"

    result = run_on_tpu(script_path, config, args=remote_args, timeout=args.timeout)

    print(result['stdout'])
    if result['stderr']:
        print("STDERR:", result['stderr'])

    return result


def main():
    parser = argparse.ArgumentParser(description="Pallas Kernel Benchmark")
    parser.add_argument("--kernel", type=str, default="matmul",
                       choices=["matmul", "int5", "exotic", "all"],
                       help="Kernel type to benchmark")
    parser.add_argument("--sizes", type=str, default="4096",
                       help="Comma-separated matrix sizes")
    parser.add_argument("--all-quant", action="store_true",
                       help="Run all quantization benchmarks")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path")
    parser.add_argument("--tpu-ip", type=str, default=None,
                       help="TPU IP for remote execution")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds for remote execution")
    parser.add_argument("--local", action="store_true",
                       help="Run locally (on TPU VM)")

    args = parser.parse_args()

    # Parse sizes
    args.sizes = [int(s.strip()) for s in args.sizes.split(",")]

    if args.local or args.tpu_ip is None:
        # Run locally (assumes we're on TPU)
        return run_local_benchmark(args)
    else:
        # Run on remote TPU
        return run_remote_benchmark(args)


if __name__ == "__main__":
    main()
