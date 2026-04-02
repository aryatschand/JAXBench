"""Evaluation harness that runs ON the TPU.

This file is SCP'd to the TPU and executed there. It:
  1. Imports the original workload and the generated Pallas workload
  2. Runs both with identical inputs
  3. Checks correctness (allclose)
  4. Benchmarks both
  5. Prints a JSON result line to stdout

Usage (on TPU):
    PJRT_DEVICE=TPU python3 eval_harness.py \
        --original /tmp/pallas_eval/original/1_Square_matrix_multiplication_.py \
        --generated /tmp/pallas_eval/generated/1_Square_matrix_multiplication_.py \
        --suite jaxkernelbench \
        --name 1_Square_matrix_multiplication_
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def benchmark_fn(fn, inputs, num_warmup=3, num_iters=20):
    """Time a jitted function. Returns list of times in ms."""
    jitted = jax.jit(fn)
    for _ in range(num_warmup):
        out = jitted(*inputs)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = jitted(*inputs)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return times, out


def check_correctness(ref_out, test_out, atol=1e-2, rtol=1e-2):
    """Compare two outputs. Returns (is_correct, max_diff)."""
    if isinstance(ref_out, (tuple, list)):
        ref_flat = jax.tree.leaves(ref_out)
        test_flat = jax.tree.leaves(test_out)
    else:
        ref_flat = [ref_out]
        test_flat = [test_out]

    if len(ref_flat) != len(test_flat):
        return False, float("inf"), "output count mismatch"

    max_diff = 0.0
    for r, t in zip(ref_flat, test_flat):
        r_np = np.array(r, dtype=np.float32)
        t_np = np.array(t, dtype=np.float32)
        if r_np.shape != t_np.shape:
            return False, float("inf"), f"shape mismatch: {r_np.shape} vs {t_np.shape}"
        diff = np.max(np.abs(r_np - t_np))
        max_diff = max(max_diff, float(diff))
        if not np.allclose(r_np, t_np, atol=atol, rtol=rtol):
            return False, max_diff, "values differ"

    return True, max_diff, "ok"


def eval_jaxkernelbench(original_path: str, generated_path: str, name: str) -> dict:
    """Evaluate a jaxkernelbench workload."""
    orig = load_module(original_path, "original")
    gen = load_module(generated_path, "generated")

    init_inputs = orig.get_init_inputs()
    orig_model = orig.Model(*init_inputs)
    gen_model = gen.Model(*init_inputs)

    inputs = orig.get_inputs()

    ref_out = jax.jit(orig_model.forward)(*inputs)
    if hasattr(ref_out, "block_until_ready"):
        ref_out.block_until_ready()

    test_out = jax.jit(gen_model.forward)(*inputs)
    if hasattr(test_out, "block_until_ready"):
        test_out.block_until_ready()

    correct, max_diff, reason = check_correctness(ref_out, test_out)

    orig_times, _ = benchmark_fn(orig_model.forward, inputs)
    gen_times, _ = benchmark_fn(gen_model.forward, inputs)

    orig_avg = float(np.mean(orig_times))
    gen_avg = float(np.mean(gen_times))

    return {
        "name": name,
        "correct": correct,
        "max_diff": round(max_diff, 6),
        "correctness_reason": reason,
        "original_ms": round(orig_avg, 4),
        "generated_ms": round(gen_avg, 4),
        "speedup": round(orig_avg / gen_avg, 3) if gen_avg > 0 else 0,
        "original_std_ms": round(float(np.std(orig_times)), 4),
        "generated_std_ms": round(float(np.std(gen_times)), 4),
        "status": "success",
    }


def eval_priority_kernel(original_path: str, generated_path: str, name: str) -> dict:
    """Evaluate a priority_kernels workload."""
    orig = load_module(original_path, "original")
    gen = load_module(generated_path, "generated")

    inputs = orig.create_inputs()

    ref_out = jax.jit(orig.workload)(*inputs)
    if hasattr(ref_out, "block_until_ready"):
        ref_out.block_until_ready()

    test_out = jax.jit(gen.workload)(*inputs)
    if hasattr(test_out, "block_until_ready"):
        test_out.block_until_ready()

    correct, max_diff, reason = check_correctness(ref_out, test_out)

    orig_times, _ = benchmark_fn(orig.workload, inputs)
    gen_times, _ = benchmark_fn(gen.workload, inputs)

    orig_avg = float(np.mean(orig_times))
    gen_avg = float(np.mean(gen_times))

    return {
        "name": name,
        "correct": correct,
        "max_diff": round(max_diff, 6),
        "correctness_reason": reason,
        "original_ms": round(orig_avg, 4),
        "generated_ms": round(gen_avg, 4),
        "speedup": round(orig_avg / gen_avg, 3) if gen_avg > 0 else 0,
        "original_std_ms": round(float(np.std(orig_times)), 4),
        "generated_std_ms": round(float(np.std(gen_times)), 4),
        "status": "success",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--suite", required=True, choices=["jaxkernelbench", "priority_kernels"])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    try:
        if args.suite == "jaxkernelbench":
            result = eval_jaxkernelbench(args.original, args.generated, args.name)
        else:
            result = eval_priority_kernel(args.original, args.generated, args.name)
    except Exception as e:
        result = {
            "name": args.name,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
