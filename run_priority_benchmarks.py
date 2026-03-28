#!/usr/bin/env python3
"""Run all priority_kernels benchmarks on TPU.

Deploys each workload folder's variants to TPU, runs them, collects results.
Handles errors gracefully and saves intermediate results.

Usage:
    GCP_CREDENTIALS_FILE=jaxbench-0e058ef95d8c.json TPU_SSH_USER=sa_113193615295475831590 \
    python run_priority_benchmarks.py --keep-tpu
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infrastructure.tpu_manager import TPUManager, SSH_KEY, SSH_USER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench")

PRIORITY_DIR = Path(__file__).resolve().parent / "priority_kernels"
REMOTE_DIR = "/tmp/jaxbench_pk"
VARIANTS = ["baseline", "optimized", "pallas"]


def scp(local, ip, remote, timeout=30):
    r = subprocess.run(
        ["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
         str(local), f"{SSH_USER}@{ip}:{remote}"],
        capture_output=True, text=True, timeout=timeout
    )
    if r.returncode != 0:
        raise RuntimeError(f"SCP {local} failed: {r.stderr[:200]}")


def discover():
    """Return [(workload, variant, path), ...]"""
    tasks = []
    for d in sorted(PRIORITY_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith(('_', '.')):
            continue
        for v in VARIANTS:
            f = d / f"{v}.py"
            if f.exists():
                tasks.append((d.name, v, f))
    return tasks


def run_one(tpu, workload, variant, timeout=300):
    """Run one variant on TPU, return parsed result dict."""
    remote = f"{REMOTE_DIR}/{workload}/{variant}.py"
    cmd = f"PJRT_DEVICE=TPU python3 {remote}"
    log.info(f"  {workload}/{variant}...")
    t0 = time.time()
    try:
        out = tpu.run_ssh(cmd, timeout=timeout)
    except Exception as e:
        log.warning(f"    SSH error: {e}")
        return {"name": f"{workload}_{variant}", "variant": variant, "status": "error", "error": str(e)[:300]}
    elapsed = time.time() - t0

    lines = [l.strip() for l in out.strip().split('\n') if l.strip()]
    for line in reversed(lines):
        try:
            r = json.loads(line)
            r["variant"] = variant
            log.info(f"    OK ({elapsed:.0f}s) {r.get('time_ms','?')}ms {r.get('tflops','?')} TFLOPS")
            return r
        except json.JSONDecodeError:
            continue
    error = out.strip()[-500:] if out.strip() else "No output"
    log.warning(f"    FAIL ({elapsed:.0f}s): {error[:150]}")
    return {"name": f"{workload}_{variant}", "variant": variant, "status": "error", "error": error}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tpu", default="v6e-1")
    p.add_argument("--tpu-name", default="jaxbench-runner")
    p.add_argument("--keep-tpu", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--workload", default=None)
    p.add_argument("--variant", default=None)
    args = p.parse_args()

    tasks = discover()
    if args.workload:
        tasks = [(w, v, p) for w, v, p in tasks if w == args.workload]
    if args.variant:
        tasks = [(w, v, p) for w, v, p in tasks if v == args.variant]
    if not tasks:
        log.error("No tasks found"); sys.exit(1)

    workloads = sorted(set(w for w, _, _ in tasks))
    log.info(f"Found {len(tasks)} tasks across {len(workloads)} workloads")

    tpu = TPUManager(tpu_name=args.tpu_name, tpu_type=args.tpu)
    try:
        log.info(f"Getting/creating TPU {args.tpu}...")
        tpu.get_or_create_tpu()
        log.info(f"TPU at {tpu.tpu_ip}")
        tpu.setup_environment()

        # Create remote dirs
        tpu.run_ssh(f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}")
        for w in workloads:
            tpu.run_ssh(f"mkdir -p {REMOTE_DIR}/{w}")
        tpu.run_ssh("pip install -q numpy", timeout=60)

        # Deploy files
        log.info("Deploying files...")
        for w, v, path in tasks:
            scp(path, tpu.tpu_ip, f"{REMOTE_DIR}/{w}/{v}.py")
        log.info(f"Deployed {len(tasks)} files")

        # Get JAX version
        jax_ver = "unknown"
        try:
            raw = tpu.run_ssh("python3 -c 'import jax; print(jax.__version__)'").strip()
            for line in raw.split('\n'):
                if line.strip() and line.strip()[0].isdigit():
                    jax_ver = line.strip(); break
        except:
            pass
        log.info(f"JAX version: {jax_ver}")

        # Run benchmarks
        log.info("=" * 60)
        log.info("RUNNING BENCHMARKS")
        log.info("=" * 60)
        results = {}
        ok = fail = 0
        for w, v, _ in tasks:
            tpu.clear_tpu_state()
            time.sleep(1)
            r = run_one(tpu, w, v, timeout=args.timeout)
            results.setdefault(w, {})[v] = r
            if r.get("status") == "success":
                ok += 1
            else:
                fail += 1
            # Save intermediate results
            _save(results, jax_ver, args.tpu, ok, fail)

        # Final save
        _save(results, jax_ver, args.tpu, ok, fail)

        # Print summary
        log.info("=" * 60)
        log.info(f"DONE: {ok}/{len(tasks)} succeeded, {fail} failed")
        log.info("=" * 60)
        print(f"\n{'Workload':<32} {'Variant':<11} {'Time (ms)':>10} {'TFLOPS':>8} {'Speedup':>8}")
        print("-" * 75)
        for w in sorted(results):
            bt = results[w].get("baseline", {}).get("time_ms")
            for v in VARIANTS:
                if v not in results[w]:
                    continue
                r = results[w][v]
                if r.get("status") == "success":
                    t = r["time_ms"]
                    tf = r.get("tflops", 0)
                    sp = f"{bt/t:.2f}x" if bt and v != "baseline" and t > 0 else "—"
                    print(f"{w:<32} {v:<11} {t:>10.2f} {tf:>8.2f} {sp:>8}")
                else:
                    print(f"{w:<32} {v:<11} {'FAIL':>10} {'':>8} {'':>8}")

    finally:
        if not args.keep_tpu:
            log.info("Deleting TPU...")
            tpu.delete_tpu()
        else:
            log.info(f"Keeping TPU {args.tpu_name}")


def _save(results, jax_ver, tpu_type, ok, fail):
    out = {
        "metadata": {
            "suite": "priority_kernels",
            "tpu_type": tpu_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jax_version": jax_ver,
            "num_succeeded": ok,
            "num_failed": fail,
        },
        "workloads": results,
    }
    path = PRIORITY_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
