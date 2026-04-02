"""Retry workloads that failed due to transient TPU init errors."""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from pallas_eval.tpu import run_ssh, scp_to_tpu, clear_tpu_state
from pallas_eval.evaluate import run_eval, setup_tpu, discover_generated

logger = logging.getLogger("pallas_eval.retry")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval JSON with errors")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=int, default=5, help="Seconds between retries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    with open(args.input) as f:
        data = json.load(f)

    tpu_init_indices = []
    for i, r in enumerate(data["results"]):
        if r.get("status") == "error" and "Unable to initialize backend" in r.get("error", ""):
            tpu_init_indices.append(i)

    logger.info(f"Found {len(tpu_init_indices)} TPU init errors to retry")
    if not tpu_init_indices:
        return

    model_key = data["results"][tpu_init_indices[0]].get("model", "gpt53")
    all_tasks = discover_generated(model_key)
    task_map = {t["name"]: t for t in all_tasks}

    setup_tpu()

    fixed = 0
    for idx in tpu_init_indices:
        r = data["results"][idx]
        name = r["name"]
        task = task_map.get(name)
        if not task:
            logger.warning(f"  Cannot find task for {name}, skipping")
            continue

        for attempt in range(args.max_retries):
            logger.info(f"  Retrying {name} (attempt {attempt+1}/{args.max_retries})...")
            time.sleep(args.sleep)
            clear_tpu_state()
            time.sleep(2)
            new_result = run_eval(task)
            if new_result.get("status") != "error" or "Unable to initialize backend" not in new_result.get("error", ""):
                data["results"][idx] = new_result
                if new_result.get("status") == "success":
                    c = "CORRECT" if new_result.get("correct") else "WRONG"
                    logger.info(f"    {c} | speedup={new_result.get('speedup', 0):.2f}x")
                else:
                    logger.info(f"    Now real error: {new_result.get('error', '?')[:80]}")
                fixed += 1
                break
        else:
            logger.warning(f"  {name}: still failing after {args.max_retries} retries")

    n_correct = sum(1 for r in data["results"] if r.get("status") == "success" and r.get("correct"))
    n_faster = sum(1 for r in data["results"] if r.get("status") == "success" and r.get("speedup", 0) > 1.0)
    n_error = sum(1 for r in data["results"] if r.get("status") == "error")
    data["metadata"] = {
        "total": len(data["results"]),
        "correct": n_correct,
        "faster": n_faster,
        "errors": n_error,
    }

    out_path = args.output or args.input
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Fixed {fixed}/{len(tpu_init_indices)} TPU init errors")
    logger.info(f"Updated results: {n_correct} correct, {n_faster} faster, {n_error} errors")
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
