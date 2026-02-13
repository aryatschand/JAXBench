#!/usr/bin/env python3
"""
Run All Real Workload Benchmarks

This script runs all extracted operators on TPU and aggregates results.

Usage:
    # Run locally on TPU
    PJRT_DEVICE=TPU python real_workloads/scripts/run_all_benchmarks.py

    # Run on remote TPU
    python real_workloads/scripts/run_all_benchmarks.py --tpu-ip <IP>
"""

import argparse
import json
import sys
import os
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def run_attention_benchmarks():
    """Run attention benchmarks."""
    from real_workloads.extracted.attention.dot_product_attention import (
        run_all_benchmarks as run_attention
    )
    return run_attention()


def run_normalization_benchmarks():
    """Run normalization benchmarks."""
    from real_workloads.extracted.normalization.rmsnorm import (
        run_all_benchmarks as run_rmsnorm
    )
    return run_rmsnorm()


def run_gemm_benchmarks():
    """Run GEMM/linear benchmarks."""
    from real_workloads.extracted.gemm.dense_projection import (
        run_all_benchmarks as run_gemm
    )
    return run_gemm()


def run_rope_benchmarks():
    """Run RoPE benchmarks."""
    from real_workloads.extracted.embeddings.rope import (
        run_all_benchmarks as run_rope
    )
    return run_rope()


def run_all_local():
    """Run all benchmarks locally."""
    print("=" * 100)
    print("REAL WORKLOADS BENCHMARK SUITE")
    print("=" * 100)
    print()

    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    except Exception as e:
        print(f"JAX error: {e}")
        return None

    print()

    all_results = {}

    # Attention
    print("\n" + "=" * 100)
    print("ATTENTION BENCHMARKS")
    print("=" * 100)
    try:
        all_results['attention'] = run_attention_benchmarks()
    except Exception as e:
        print(f"Attention benchmarks failed: {e}")
        all_results['attention'] = {'error': str(e)}

    # Normalization
    print("\n" + "=" * 100)
    print("NORMALIZATION BENCHMARKS")
    print("=" * 100)
    try:
        all_results['normalization'] = run_normalization_benchmarks()
    except Exception as e:
        print(f"Normalization benchmarks failed: {e}")
        all_results['normalization'] = {'error': str(e)}

    # GEMM
    print("\n" + "=" * 100)
    print("GEMM/LINEAR BENCHMARKS")
    print("=" * 100)
    try:
        all_results['gemm'] = run_gemm_benchmarks()
    except Exception as e:
        print(f"GEMM benchmarks failed: {e}")
        all_results['gemm'] = {'error': str(e)}

    # RoPE
    print("\n" + "=" * 100)
    print("ROPE BENCHMARKS")
    print("=" * 100)
    try:
        all_results['rope'] = run_rope_benchmarks()
    except Exception as e:
        print(f"RoPE benchmarks failed: {e}")
        all_results['rope'] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_benchmarks = 0
    successful = 0

    for category, results in all_results.items():
        if isinstance(results, list):
            count = len(results)
            total_benchmarks += count
            successful += count
            print(f"{category}: {count} benchmarks completed")
        elif isinstance(results, dict) and 'error' in results:
            print(f"{category}: FAILED - {results['error'][:50]}")
        else:
            print(f"{category}: Unknown result format")

    print()
    print(f"Total: {successful}/{total_benchmarks} benchmarks successful")

    return all_results


def run_on_tpu(tpu_ip: str, timeout: int = 600):
    """Copy script to TPU and run."""
    import subprocess

    key_path = os.path.expanduser("~/.ssh/id_rsa_tpu")
    user = "REDACTED_SSH_USER"

    # Copy the entire real_workloads directory
    print(f"Copying real_workloads to TPU {tpu_ip}...")

    # Create a tar of the directory
    tar_cmd = "tar -czf /tmp/real_workloads.tar.gz -C . real_workloads"
    subprocess.run(tar_cmd, shell=True, check=True)

    # SCP the tarball
    scp_cmd = f"scp -i {key_path} -o StrictHostKeyChecking=no /tmp/real_workloads.tar.gz {user}@{tpu_ip}:~/"
    result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SCP failed: {result.stderr}")
        return None

    # Extract and run on TPU
    ssh_cmd = f"""ssh -i {key_path} -o StrictHostKeyChecking=no {user}@{tpu_ip} '
        cd ~ &&
        rm -rf real_workloads &&
        tar -xzf real_workloads.tar.gz &&
        PJRT_DEVICE=TPU python3 real_workloads/scripts/run_all_benchmarks.py --local
    '"""

    print("Running benchmarks on TPU...")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run real workload benchmarks")
    parser.add_argument("--tpu-ip", type=str, help="TPU IP for remote execution")
    parser.add_argument("--local", action="store_true", help="Run locally (on TPU VM)")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for remote execution")

    args = parser.parse_args()

    if args.local or args.tpu_ip is None:
        results = run_all_local()

        if args.output and results:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': results,
                }, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
    else:
        result = run_on_tpu(args.tpu_ip, args.timeout)
        return result


if __name__ == "__main__":
    main()
