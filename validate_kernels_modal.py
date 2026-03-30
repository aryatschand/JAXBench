"""Validate all priority_kernels on Modal Sandboxes with GPU.

Uses Modal Sandbox API to upload and run each kernel variant.
Checks: import, JIT, execution, shapes, NaN-free, correctness.

Usage:
    python validate_kernels_modal.py
"""
import modal
import json
import os
from pathlib import Path

WORKLOADS = [
    'cross_entropy', 'flash_attention', 'flex_attention', 'gemm',
    'gqa_attention', 'mamba2_ssd', 'megablox_gmm', 'mla_attention',
    'paged_attention', 'ragged_dot', 'ragged_paged_attention',
    'retnet_retention', 'rms_norm', 'sparse_attention', 'sparse_moe',
    'swiglu_mlp', 'triangle_multiplication',
]

OPTIMIZED = [
    'flash_attention', 'flex_attention', 'gqa_attention', 'mla_attention',
    'paged_attention', 'ragged_dot', 'sparse_attention', 'sparse_moe',
]

VALIDATE_SCRIPT = '''
import sys, json, importlib, traceback
sys.path.insert(0, "/root")
import jax
import jax.numpy as jnp
import numpy as np

workload = sys.argv[1]
variant = sys.argv[2]

result = {"workload": workload, "variant": variant, "status": "error"}

try:
    mod = importlib.import_module(f"priority_kernels.{workload}.{variant}")
    result["import"] = True
    result["config_name"] = mod.CONFIG.get("name", "?")

    skip_jit = getattr(mod, "_skip_jit", False)
    inputs = mod.create_inputs(dtype=jnp.bfloat16)
    result["create_inputs"] = True
    result["input_shapes"] = [list(x.shape) if hasattr(x, "shape") else str(type(x)) for x in inputs]

    if skip_jit:
        result["jit"] = "skip"
        out = mod.workload(*inputs)
    else:
        fn = jax.jit(mod.workload)
        fn.lower(*inputs)  # verify compilation
        result["jit"] = True
        out = fn(*inputs)
        out.block_until_ready()

    result["run"] = True
    if hasattr(out, "shape"):
        result["shape"] = list(out.shape)
        result["dtype"] = str(out.dtype)
        result["nan_free"] = bool(not jnp.any(jnp.isnan(out)))
    else:
        result["shape"] = "scalar"
        result["nan_free"] = True

    result["status"] = "ok" if result["nan_free"] else "nan"
except Exception as e:
    result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    result["traceback"] = traceback.format_exc()[-500:]

print("RESULT:" + json.dumps(result))
'''

CORRECTNESS_SCRIPT = '''
import sys, json, importlib, traceback
sys.path.insert(0, "/root")
import jax
import jax.numpy as jnp
import numpy as np

workload = sys.argv[1]
result = {"workload": workload, "check": "correctness", "status": "error"}

try:
    base = importlib.import_module(f"priority_kernels.{workload}.baseline")
    opt = importlib.import_module(f"priority_kernels.{workload}.optimized")

    inputs_b = base.create_inputs(dtype=jnp.float32)
    inputs_o = opt.create_inputs(dtype=jnp.float32)

    skip_b = getattr(base, "_skip_jit", False)
    skip_o = getattr(opt, "_skip_jit", False)

    out_b = base.workload(*inputs_b) if skip_b else jax.jit(base.workload)(*inputs_b)
    if hasattr(out_b, "block_until_ready"): out_b.block_until_ready()

    out_o = opt.workload(*inputs_o) if skip_o else jax.jit(opt.workload)(*inputs_o)
    if hasattr(out_o, "block_until_ready"): out_o.block_until_ready()

    b = np.array(out_b).flatten()
    o = np.array(out_o).flatten()

    if b.shape != o.shape:
        result["error"] = f"shape mismatch: {out_b.shape} vs {out_o.shape}"
    else:
        max_diff = float(np.max(np.abs(b - o)))
        mean_diff = float(np.mean(np.abs(b - o)))
        result["max_diff"] = round(max_diff, 6)
        result["mean_diff"] = round(mean_diff, 6)
        result["match"] = max_diff < 0.5
        result["status"] = "ok"
        result["shapes"] = {"baseline": list(out_b.shape), "optimized": list(out_o.shape)}
except Exception as e:
    result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

print("RESULT:" + json.dumps(result))
'''

jax_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jax[cuda12]", "numpy")
)

def upload_kernels(sb):
    """Upload all priority_kernels files to sandbox."""
    pk_dir = Path("priority_kernels")
    sb.mkdir("/root/priority_kernels", parents=True)

    # Create __init__.py files
    sb.open("/root/priority_kernels/__init__.py", "w").write("")

    for d in sorted(pk_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(('_', '.')):
            continue
        sb.mkdir(f"/root/priority_kernels/{d.name}", parents=True)
        sb.open(f"/root/priority_kernels/{d.name}/__init__.py", "w").write("")
        for f in d.glob("*.py"):
            content = f.read_text()
            sb.open(f"/root/priority_kernels/{d.name}/{f.name}", "w").write(content)


def run_in_sandbox(sb, script, args):
    """Write script and run it, return parsed result."""
    sb.open("/root/run_validate.py", "w").write(script)
    proc = sb.exec("python3", "/root/run_validate.py", *args)
    stdout = proc.stdout.read()
    stderr = proc.stderr.read()
    proc.wait()

    # Parse result from stdout
    for line in stdout.strip().split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    return {"error": f"No result. stdout={stdout[-200:]}, stderr={stderr[-200:]}"}


def main():
    app = modal.App.lookup("jaxbench-validate", create_if_missing=True)

    # Build task list
    tasks = [(w, "baseline") for w in WORKLOADS]
    tasks += [(w, "optimized") for w in OPTIMIZED]

    print(f"Validating {len(tasks)} kernel variants on Modal GPU...")
    print()

    results = []
    correctness = []

    # Create sandbox
    print("Creating sandbox...")
    sb = modal.Sandbox.create(
        image=jax_image,
        gpu="T4",
        timeout=900,
        app=app,
    )

    try:
        print("Uploading kernels...")
        upload_kernels(sb)
        print("Kernels uploaded.\n")

        # Validate each variant
        print(f"{'Workload':<30} {'Variant':<11} {'JIT':>5} {'Run':>5} {'NaN-free':>9} {'Shape':<25}")
        print("-" * 90)

        for w, v in tasks:
            r = run_in_sandbox(sb, VALIDATE_SCRIPT, [w, v])
            results.append(r)

            jit = "OK" if r.get("jit") is True else ("skip" if r.get("jit") == "skip" else "FAIL")
            run = "OK" if r.get("run") else "FAIL"
            nan = "OK" if r.get("nan_free") else "FAIL"
            shape = str(r.get("shape", "—"))
            fail = "" if r.get("status") == "ok" else f"  <<<{r.get('error', '')[:40]}"

            print(f"{w:<30} {v:<11} {jit:>5} {run:>5} {nan:>9} {shape:<25}{fail}")

        # Correctness checks
        print()
        print("=" * 80)
        print("CORRECTNESS: baseline vs optimized")
        print("=" * 80)
        print(f"{'Workload':<30} {'Max Diff':>12} {'Mean Diff':>12} {'Match':>8}")
        print("-" * 65)

        for w in OPTIMIZED:
            r = run_in_sandbox(sb, CORRECTNESS_SCRIPT, [w])
            correctness.append(r)

            if r.get("error"):
                print(f"{w:<30} {'ERROR':>12} {r['error'][:35]}")
            else:
                md = r.get("max_diff", -1)
                mn = r.get("mean_diff", -1)
                match = "YES" if r.get("match") else "NO"
                print(f"{w:<30} {md:>12.6f} {mn:>12.6f} {match:>8}")

    finally:
        sb.terminate()

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nSUMMARY: {ok}/{total} variants passed")

    failures = [r for r in results if r.get("status") != "ok"]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f['workload']}/{f['variant']}: {f.get('error', 'unknown')[:80]}")

    # Save
    all_results = {"validation": results, "correctness": correctness}
    with open("priority_kernels/validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to priority_kernels/validation_results.json")


if __name__ == "__main__":
    main()
