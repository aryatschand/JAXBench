"""
Benchmark runner for priority_kernels workloads on TPU.

Discovers workload folders, each containing baseline.py (and optionally
optimized.py, pallas.py), copies them to a TPU VM, runs each variant,
collects JSON results, and saves to priority_kernels/results.json.

Usage:
    python -m priority_kernels.run_benchmarks --tpu v6e-1 --keep-tpu
    python -m priority_kernels.run_benchmarks --tpu v6e-1 --workload flash_attention --variant optimized
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.tpu_manager import TPUManager, SSH_KEY, SSH_USER

logger = logging.getLogger("jaxbench.priority_kernels")

VARIANTS = ["baseline", "optimized", "pallas"]
WORKLOADS_DIR = Path(__file__).resolve().parent
REMOTE_DIR = "/tmp/jaxbench_priority_kernels"


def scp_file(local_path: str, remote_ip: str, remote_path: str, timeout: int = 30):
    """Copy a file to the TPU VM via SCP."""
    cmd = [
        "scp",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        local_path,
        f"{SSH_USER}@{remote_ip}:{remote_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def run_workload_on_tpu(tpu: TPUManager, workload: str, variant: str,
                        timeout: int = 300) -> dict:
    """Run a single workload variant on the TPU and parse JSON output."""
    remote_file = f"{REMOTE_DIR}/{workload}/{variant}.py"
    cmd = f"PJRT_DEVICE=TPU python3 {remote_file}"

    logger.info(f"  Running {workload}/{variant}...")
    start = time.time()
    output = tpu.run_ssh(cmd, timeout=timeout)
    elapsed = time.time() - start

    # Try to parse JSON from the last non-empty line of output
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    for line in reversed(lines):
        try:
            result = json.loads(line)
            result['variant'] = variant
            logger.info(f"    OK ({elapsed:.1f}s) — {result.get('time_ms', '?')}ms")
            return result
        except json.JSONDecodeError:
            continue

    # Failed to parse
    error_msg = output.strip()[-500:] if output.strip() else "No output"
    logger.warning(f"    FAILED ({elapsed:.1f}s): {error_msg[:100]}")
    return {
        'name': f"{workload}_{variant}",
        'variant': variant,
        'status': 'error',
        'error': error_msg,
    }


def discover_workloads(workload_filter=None, variant_filter=None) -> list:
    """Discover workload folders and their variants.

    Returns list of (workload_name, variant_name, local_path) tuples.
    """
    discovered = []
    for entry in sorted(WORKLOADS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith(('_', '.')):
            continue
        if workload_filter and entry.name != workload_filter:
            continue
        for variant in VARIANTS:
            variant_file = entry / f"{variant}.py"
            if variant_file.exists():
                if variant_filter and variant != variant_filter:
                    continue
                discovered.append((entry.name, variant, str(variant_file)))
    return discovered


def main():
    parser = argparse.ArgumentParser(description="Run priority kernel benchmarks on TPU")
    parser.add_argument("--tpu", default="v6e-1", help="TPU type (default: v6e-1)")
    parser.add_argument("--tpu-name", default="jaxbench-runner", help="TPU VM name")
    parser.add_argument("--keep-tpu", action="store_true", help="Keep TPU after benchmarks")
    parser.add_argument("--timeout", type=int, default=300, help="Per-workload timeout (s)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--workload", default=None, help="Run only this workload folder")
    parser.add_argument("--variant", default=None, choices=VARIANTS,
                        help="Run only this variant (baseline/optimized/pallas)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Discover workloads
    tasks = discover_workloads(
        workload_filter=args.workload,
        variant_filter=args.variant,
    )

    if not tasks:
        logger.error("No workloads found matching filters.")
        sys.exit(1)

    workload_names = sorted(set(w for w, _, _ in tasks))
    logger.info(f"Found {len(tasks)} tasks across {len(workload_names)} workloads")

    # Allocate TPU
    tpu = TPUManager(tpu_name=args.tpu_name, tpu_type=args.tpu)
    try:
        logger.info(f"Getting/creating TPU {args.tpu}...")
        tpu.get_or_create_tpu()
        tpu.setup_environment()

        # Create remote directories and install numpy
        tpu.run_ssh(f"mkdir -p {REMOTE_DIR}")
        for wname in workload_names:
            tpu.run_ssh(f"mkdir -p {REMOTE_DIR}/{wname}")
        tpu.run_ssh("pip install -q numpy", timeout=60)

        # Copy all workload files to TPU
        logger.info("Copying workload files to TPU...")
        for wname, variant, local_path in tasks:
            remote_path = f"{REMOTE_DIR}/{wname}/{variant}.py"
            scp_file(local_path, tpu.tpu_ip, remote_path)
        logger.info(f"  Copied {len(tasks)} files")

        # Clear TPU state before benchmarking
        tpu.clear_tpu_state()
        time.sleep(2)

        # Get JAX version from TPU
        jax_version_raw = tpu.run_ssh("python3 -c 'import jax; print(jax.__version__)'").strip()
        jax_version = "unknown"
        for line in jax_version_raw.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                jax_version = line
                break

        # Run each workload variant
        logger.info("=" * 60)
        logger.info("RUNNING PRIORITY KERNEL BENCHMARKS")
        logger.info("=" * 60)
        results = {}  # workload -> {variant -> result}
        num_success = 0
        num_failed = 0

        for wname, variant, _ in tasks:
            tpu.clear_tpu_state()
            time.sleep(1)

            result = run_workload_on_tpu(tpu, wname, variant, timeout=args.timeout)

            if wname not in results:
                results[wname] = {}
            results[wname][variant] = result

            if result.get('status') == 'success':
                num_success += 1
            else:
                num_failed += 1

        # Build output JSON
        output = {
            'metadata': {
                'suite': 'priority_kernels',
                'description': f'{len(workload_names)} workloads with {len(tasks)} total variants',
                'tpu_type': args.tpu,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'jax_version': jax_version,
                'num_workloads': len(workload_names),
                'num_variants': len(tasks),
                'num_succeeded': num_success,
                'num_failed': num_failed,
            },
            'workloads': results,
        }

        # Save results
        output_path = args.output or str(WORKLOADS_DIR / "results.json")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"RESULTS: {num_success}/{len(tasks)} succeeded, {num_failed} failed")
        logger.info(f"Saved to: {output_path}")
        logger.info("=" * 60)

        # Print summary table
        print(f"\n{'Workload':<35} {'Variant':<12} {'Status':<8} {'Time (ms)':>10} {'TFLOPS':>8}")
        print("-" * 78)
        for wname in sorted(results.keys()):
            baseline_time = None
            for variant in VARIANTS:
                if variant not in results[wname]:
                    continue
                r = results[wname][variant]
                status = r.get('status', 'error')
                if status == 'success':
                    t = r['time_ms']
                    tflops = r.get('tflops', 0)
                    if variant == 'baseline':
                        baseline_time = t
                    speedup = ""
                    if baseline_time and variant != 'baseline' and t > 0:
                        speedup = f" ({baseline_time / t:.2f}x)"
                    print(f"{wname:<35} {variant:<12} {'OK':<8} {t:>10.2f} {tflops:>8.2f}{speedup}")
                else:
                    err = r.get('error', '')[:25]
                    print(f"{wname:<35} {variant:<12} {'FAIL':<8} {'':>10} {err}")

    finally:
        if not args.keep_tpu:
            logger.info("Deleting TPU...")
            tpu.delete_tpu()
        else:
            logger.info(f"Keeping TPU {args.tpu_name} (--keep-tpu)")


if __name__ == '__main__':
    main()
