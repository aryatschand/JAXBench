"""
Benchmark runner for real workloads on TPU.

Discovers all workload files from model subdirectories, copies them to a TPU VM,
runs each one, collects JSON results, and saves to results/real_workload_benchmarks.json.

Usage:
    python -m real_workloads.run_benchmarks --tpu v6e-1 --keep-tpu
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

logger = logging.getLogger("jaxbench.benchmarks")

# Workload files organized by model directory
WORKLOAD_MAP = {
    "llama3": [
        "llama3_8b_gqa.py",
        "llama3_70b_gqa.py",
        "llama3_405b_gqa.py",
        "llama3_8b_swiglu.py",
        "llama3_70b_swiglu.py",
        "llama3_405b_swiglu.py",
        "llama3_8b_rope.py",
        "llama3_70b_rope.py",
        "llama3_8b_rmsnorm.py",
        "llama3_70b_rmsnorm.py",
        "llama3_8b_token_embed.py",
    ],
    "gemma3": [
        "gemma3_4b_sliding_window_attn.py",
        "gemma3_12b_sliding_window_attn.py",
        "gemma3_27b_sliding_window_attn.py",
        "gemma3_4b_global_attn.py",
        "gemma3_12b_global_attn.py",
        "gemma3_27b_global_attn.py",
    ],
    "mixtral": [
        "mixtral_8x7b_moe.py",
        "mixtral_8x22b_moe.py",
    ],
    "deepseek_v3": [
        "deepseek_v3_mla.py",
        "deepseek_v3_yarn_rope.py",
        "deepseek_v3_moe.py",
    ],
    "qwen3": [
        "qwen3_8b_gqa.py",
        "qwen3_14b_gqa.py",
        "qwen3_8b_swiglu.py",
        "qwen3_moe_30b_moe.py",
    ],
    "llama4": [
        "llama4_scout_gqa.py",
        "llama4_scout_moe.py",
        "llama4_maverick_moe.py",
        "llama4_scout_rope.py",
    ],
}

WORKLOADS_DIR = Path(__file__).resolve().parent
REMOTE_DIR = "/tmp/jaxbench_workloads"


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


def run_workload_on_tpu(tpu: TPUManager, filename: str, timeout: int = 300) -> dict:
    """Run a single workload file on the TPU and parse JSON output."""
    remote_file = f"{REMOTE_DIR}/{filename}"
    cmd = f"PJRT_DEVICE=TPU python3 {remote_file}"

    logger.info(f"  Running {filename}...")
    start = time.time()
    output = tpu.run_ssh(cmd, timeout=timeout)
    elapsed = time.time() - start

    # Try to parse JSON from the last non-empty line of output
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    for line in reversed(lines):
        try:
            result = json.loads(line)
            logger.info(f"    OK ({elapsed:.1f}s) — {result.get('time_ms', '?')}ms")
            return result
        except json.JSONDecodeError:
            continue

    # Failed to parse
    error_msg = output.strip()[-500:] if output.strip() else "No output"
    logger.warning(f"    FAILED ({elapsed:.1f}s): {error_msg[:100]}")
    name = filename.replace('.py', '')
    return {
        'name': name,
        'status': 'error',
        'error': error_msg,
    }


def discover_workloads(model_filter=None) -> list:
    """Return list of (model_dir, filename) tuples for workloads on disk."""
    available = []
    for model_dir, files in WORKLOAD_MAP.items():
        if model_filter and model_dir != model_filter:
            continue
        for f in files:
            local_path = WORKLOADS_DIR / model_dir / f
            if local_path.exists():
                available.append((model_dir, f))
            else:
                logger.warning(f"Workload file not found: {model_dir}/{f}")
    return available


def main():
    parser = argparse.ArgumentParser(description="Run real workload benchmarks on TPU")
    parser.add_argument("--tpu", default="v6e-1", help="TPU type (default: v6e-1)")
    parser.add_argument("--tpu-name", default="jaxbench-runner", help="TPU VM name")
    parser.add_argument("--keep-tpu", action="store_true", help="Keep TPU after benchmarks")
    parser.add_argument("--timeout", type=int, default=300, help="Per-workload timeout (s)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--workload", default=None, help="Run only this workload (filename without .py)")
    parser.add_argument("--model", default=None, help="Run only workloads for this model (e.g., llama3)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Discover workloads
    workloads = discover_workloads(model_filter=args.model)

    if args.workload:
        target = f"{args.workload}.py"
        workloads = [(m, f) for m, f in workloads if f == target]
        if not workloads:
            logger.error(f"Workload {target} not found.")
            sys.exit(1)

    logger.info(f"Found {len(workloads)} workloads to benchmark")

    # Allocate TPU
    tpu = TPUManager(tpu_name=args.tpu_name, tpu_type=args.tpu)
    try:
        logger.info(f"Getting/creating TPU {args.tpu}...")
        tpu.get_or_create_tpu()
        tpu.setup_environment()

        # Create remote directory and install numpy
        tpu.run_ssh(f"mkdir -p {REMOTE_DIR}")
        tpu.run_ssh("pip install -q numpy", timeout=60)

        # Copy all workload files to TPU
        logger.info("Copying workload files to TPU...")
        for model_dir, f in workloads:
            local_path = str(WORKLOADS_DIR / model_dir / f)
            scp_file(local_path, tpu.tpu_ip, f"{REMOTE_DIR}/{f}")
        logger.info(f"  Copied {len(workloads)} files")

        # Clear TPU state before benchmarking
        tpu.clear_tpu_state()
        time.sleep(2)

        # Get JAX version from TPU
        jax_version_raw = tpu.run_ssh("python3 -c 'import jax; print(jax.__version__)'").strip()
        # Filter out SSH warnings — find line matching version pattern
        jax_version = "unknown"
        for line in jax_version_raw.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                jax_version = line
                break

        # Run each workload
        logger.info("=" * 60)
        logger.info("RUNNING BENCHMARKS")
        logger.info("=" * 60)
        results = []
        num_success = 0
        num_failed = 0

        for model_dir, f in workloads:
            # Clear TPU state between workloads to avoid interference
            tpu.clear_tpu_state()
            time.sleep(1)

            result = run_workload_on_tpu(tpu, f, timeout=args.timeout)
            results.append(result)

            if result.get('status') == 'success':
                num_success += 1
            else:
                num_failed += 1

        # Build output JSON
        output = {
            'metadata': {
                'tpu_type': args.tpu,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'jax_version': jax_version,
                'num_workloads': len(workloads),
                'num_succeeded': num_success,
                'num_failed': num_failed,
            },
            'workloads': results,
        }

        # Save results
        output_path = args.output or str(WORKLOADS_DIR / "results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"RESULTS: {num_success}/{len(workloads)} succeeded, {num_failed} failed")
        logger.info(f"Saved to: {output_path}")
        logger.info("=" * 60)

        # Print summary table
        print(f"\n{'Name':<40} {'Status':<8} {'Time (ms)':>10} {'TFLOPS':>8}")
        print("-" * 70)
        for r in results:
            status = r.get('status', 'error')
            name = r.get('name', '?')
            if status == 'success':
                print(f"{name:<40} {'OK':<8} {r['time_ms']:>10.2f} {r.get('tflops', 0):>8.2f}")
            else:
                err = r.get('error', '')[:30]
                print(f"{name:<40} {'FAIL':<8} {'':>10} {err}")

    finally:
        if not args.keep_tpu:
            logger.info("Deleting TPU...")
            tpu.delete_tpu()
        else:
            logger.info(f"Keeping TPU {args.tpu_name} (--keep-tpu)")


if __name__ == '__main__':
    main()
