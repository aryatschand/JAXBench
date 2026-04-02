"""Validate that all 217 original JAX workloads run correctly on TPU.

SCPs each workload to the TPU, runs it, and records pass/fail.

Usage:
    python -m pallas_eval.validate_baselines
    python -m pallas_eval.validate_baselines --suite jaxkernelbench
    python -m pallas_eval.validate_baselines --suite priority_kernels
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from pallas_eval.tpu import run_ssh, scp_to_tpu, clear_tpu_state

logger = logging.getLogger("pallas_eval.validate_baselines")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PALLAS_EVAL_DIR = Path(__file__).resolve().parent
JAXKERNELBENCH_DIR = PROJECT_ROOT / "jaxkernelbench"
PRIORITY_DIR = PROJECT_ROOT / "priority_kernels"
REMOTE_BASE = "/tmp/pallas_eval"

BASELINE_HARNESS = """\
import importlib.util, json, sys, time, traceback
import jax, jax.numpy as jnp, numpy as np

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

path = sys.argv[1]
suite = sys.argv[2]
name = sys.argv[3]

try:
    mod = load_module(path, "workload")
    if suite == "jaxkernelbench":
        init_inputs = mod.get_init_inputs()
        model = mod.Model(*init_inputs)
        inputs = mod.get_inputs()
        out = jax.jit(model.forward)(*inputs)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        # Quick benchmark (5 warmup + 10 iters)
        jitted = jax.jit(model.forward)
        for _ in range(5):
            o = jitted(*inputs)
            if hasattr(o, "block_until_ready"): o.block_until_ready()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            o = jitted(*inputs)
            if hasattr(o, "block_until_ready"): o.block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)
    else:
        inputs = mod.create_inputs()
        out = jax.jit(mod.workload)(*inputs)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        jitted = jax.jit(mod.workload)
        for _ in range(5):
            o = jitted(*inputs)
            if hasattr(o, "block_until_ready"): o.block_until_ready()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            o = jitted(*inputs)
            if hasattr(o, "block_until_ready"): o.block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)

    out_shape = str(jax.tree.map(lambda x: (x.shape, x.dtype), out)) if hasattr(out, 'shape') else str(type(out))
    print(json.dumps({
        "name": name, "status": "pass",
        "avg_ms": round(float(np.mean(times)), 4),
        "std_ms": round(float(np.std(times)), 4),
        "output_shape": out_shape[:200],
    }))
except Exception as e:
    print(json.dumps({
        "name": name, "status": "fail",
        "error": str(e)[:500],
        "traceback": traceback.format_exc()[-500:],
    }))
"""


def discover_workloads(suite_filter=None):
    workloads = []
    if suite_filter in (None, "jaxkernelbench"):
        for level in ["level1", "level2"]:
            level_dir = JAXKERNELBENCH_DIR / level
            if not level_dir.exists():
                continue
            for f in sorted(level_dir.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                workloads.append({
                    "name": f.stem, "path": f, "level": level,
                    "suite": "jaxkernelbench",
                })
    if suite_filter in (None, "priority_kernels"):
        for entry in sorted(PRIORITY_DIR.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            baseline = entry / "baseline.py"
            if baseline.exists():
                workloads.append({
                    "name": entry.name, "path": baseline, "level": None,
                    "suite": "priority_kernels",
                })
    return workloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["jaxkernelbench", "priority_kernels"], default=None)
    parser.add_argument("--output", default="pallas_eval/results/baseline_validation.json")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    workloads = discover_workloads(args.suite)
    logger.info(f"Found {len(workloads)} workloads to validate")

    # Setup TPU
    run_ssh(f"mkdir -p {REMOTE_BASE}/baselines", timeout=15)
    harness_path = f"{REMOTE_BASE}/baseline_harness.py"
    tmp_local = "/tmp/baseline_harness.py"
    with open(tmp_local, "w") as f:
        f.write(BASELINE_HARNESS)
    scp_to_tpu(tmp_local, harness_path)

    results = []
    n_pass = 0
    n_fail = 0

    for i, w in enumerate(workloads, 1):
        name = w["name"]
        suite = w["suite"]
        remote_path = f"{REMOTE_BASE}/baselines/{name}.py"

        scp_to_tpu(str(w["path"]), remote_path)

        cmd = f"PJRT_DEVICE=TPU python3 {harness_path} {remote_path} {suite} {name}"
        logger.info(f"[{i}/{len(workloads)}] {suite}/{name}")

        max_attempts = 3
        result = None
        for attempt in range(max_attempts):
            clear_tpu_state()
            time.sleep(3)
            try:
                output = run_ssh(cmd, timeout=args.timeout)
                lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
                parsed = None
                for line in reversed(lines):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

                if parsed is None:
                    parsed = {"name": name, "status": "fail", "error": "no JSON output"}

            except Exception as e:
                parsed = {"name": name, "status": "fail", "error": str(e)[:500]}

            if "Unable to initialize backend" in parsed.get("error", ""):
                logger.info(f"  TPU init error, retry {attempt+1}/{max_attempts}...")
                time.sleep(5)
                continue

            result = parsed
            break

        if result is None:
            result = parsed

        result["suite"] = suite
        result["level"] = w.get("level")
        results.append(result)

        if result["status"] == "pass":
            n_pass += 1
            logger.info(f"  PASS  {result.get('avg_ms', '?')}ms")
        else:
            n_fail += 1
            logger.info(f"  FAIL  {result.get('error', '?')[:80]}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "metadata": {"total": len(results), "pass": n_pass, "fail": n_fail},
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"BASELINE VALIDATION: {n_pass}/{len(results)} pass, {n_fail} fail")
    logger.info(f"Saved to {args.output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
