"""Stage 2: Evaluate generated Pallas kernels on TPU.

SCPs original + generated code to the TPU, runs eval_harness.py for each,
collects results.

Usage:
    python -m pallas_eval.evaluate                          # all
    python -m pallas_eval.evaluate --model gpt53            # one model
    python -m pallas_eval.evaluate --suite jaxkernelbench   # one suite
    python -m pallas_eval.evaluate --workload gemm          # one workload
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

logger = logging.getLogger("pallas_eval.evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PALLAS_EVAL_DIR = Path(__file__).resolve().parent
GENERATED_DIR = PALLAS_EVAL_DIR / "generated"
RESULTS_DIR = PALLAS_EVAL_DIR / "results"
JAXKERNELBENCH_DIR = PROJECT_ROOT / "jaxkernelbench"
PRIORITY_DIR = PROJECT_ROOT / "priority_kernels"

REMOTE_BASE = "/tmp/pallas_eval"
REMOTE_HARNESS = f"{REMOTE_BASE}/eval_harness.py"

MODEL_KEYS = ["gpt53", "gemini3"]


def discover_generated(model_key: str, suite_filter: str | None = None,
                       workload_filter: str | None = None) -> list[dict]:
    """Find generated files and pair them with originals."""
    tasks = []
    model_dir = GENERATED_DIR / model_key

    if not model_dir.exists():
        return tasks

    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name.startswith("jaxkernelbench_"):
            suite = "jaxkernelbench"
            level = subdir.name.split("_", 1)[1]  # level1 or level2
        elif subdir.name == "priority_kernels":
            suite = "priority_kernels"
            level = None
        else:
            continue

        if suite_filter and suite != suite_filter:
            continue

        for gen_file in sorted(subdir.glob("*.py")):
            name = gen_file.stem
            if workload_filter and name != workload_filter:
                continue

            if suite == "jaxkernelbench":
                orig_file = JAXKERNELBENCH_DIR / level / gen_file.name
            else:
                orig_file = PRIORITY_DIR / name / "baseline.py"

            if not orig_file.exists():
                logger.warning(f"Original not found for {name}, skipping")
                continue

            tasks.append({
                "name": name,
                "suite": suite,
                "level": level,
                "model": model_key,
                "original_path": orig_file,
                "generated_path": gen_file,
            })

    return tasks


def setup_tpu():
    """Create remote dirs and copy harness."""
    logger.info("Setting up TPU remote directories...")
    run_ssh(f"mkdir -p {REMOTE_BASE}/originals {REMOTE_BASE}/generated", timeout=15)
    scp_to_tpu(str(PALLAS_EVAL_DIR / "eval_harness.py"), REMOTE_HARNESS)
    run_ssh("pip install -q numpy 2>/dev/null", timeout=60)


def run_eval(task: dict, timeout: int = 300) -> dict:
    """SCP files, run harness on TPU, parse result."""
    name = task["name"]
    model = task["model"]
    suite = task["suite"]

    remote_orig = f"{REMOTE_BASE}/originals/{name}_original.py"
    remote_gen = f"{REMOTE_BASE}/generated/{name}_{model}.py"

    scp_to_tpu(str(task["original_path"]), remote_orig)
    scp_to_tpu(str(task["generated_path"]), remote_gen)

    cmd = (
        f"PJRT_DEVICE=TPU python3 {REMOTE_HARNESS} "
        f"--original {remote_orig} --generated {remote_gen} "
        f"--suite {suite} --name {name}"
    )

    clear_tpu_state()
    time.sleep(1)

    logger.info(f"  Running {name} ({model})...")
    t0 = time.time()
    output = run_ssh(cmd, timeout=timeout)
    elapsed = time.time() - t0

    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        try:
            result = json.loads(line)
            result["model"] = model
            result["suite"] = suite
            result["level"] = task.get("level")
            result["eval_time_s"] = round(elapsed, 1)
            return result
        except json.JSONDecodeError:
            continue

    error_msg = output.strip()[-500:] if output.strip() else "No output"
    logger.warning(f"  FAILED ({elapsed:.1f}s): {error_msg[:100]}")
    return {
        "name": name,
        "model": model,
        "suite": suite,
        "level": task.get("level"),
        "status": "error",
        "error": error_msg,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated Pallas kernels on TPU")
    parser.add_argument("--model", choices=MODEL_KEYS, default=None)
    parser.add_argument("--suite", choices=["jaxkernelbench", "priority_kernels"], default=None)
    parser.add_argument("--workload", default=None, help="Evaluate only this workload name")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    models = [args.model] if args.model else MODEL_KEYS
    all_tasks = []
    for m in models:
        all_tasks.extend(discover_generated(m, suite_filter=args.suite, workload_filter=args.workload))

    if not all_tasks:
        logger.error("No generated files found. Run generate.py first.")
        return

    logger.info(f"Found {len(all_tasks)} evaluation tasks")
    setup_tpu()

    results = []
    n_correct = 0
    n_faster = 0
    n_error = 0

    for i, task in enumerate(all_tasks, 1):
        logger.info(f"[{i}/{len(all_tasks)}] {task['suite']}/{task['name']} ({task['model']})")
        result = run_eval(task, timeout=args.timeout)
        results.append(result)

        if result.get("status") == "success":
            c = "CORRECT" if result.get("correct") else "WRONG"
            s = result.get("speedup", 0)
            tag = "FASTER" if s > 1 else "SLOWER"
            logger.info(f"    {c} | {tag} {s:.2f}x | orig={result.get('original_ms',0):.2f}ms gen={result.get('generated_ms',0):.2f}ms")
            if result.get("correct"):
                n_correct += 1
            if s > 1:
                n_faster += 1
        else:
            n_error += 1
            logger.info(f"    ERROR: {result.get('error', '?')[:80]}")

    # Save results
    output_path = args.output or str(RESULTS_DIR / "eval_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    output = {
        "metadata": {
            "total": len(results),
            "correct": n_correct,
            "faster": n_faster,
            "errors": n_error,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {len(results)} total | {n_correct} correct | {n_faster} faster | {n_error} errors")
    logger.info(f"Saved to {output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
