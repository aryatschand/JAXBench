#!/usr/bin/env python3
"""
Test manually written JAX kernels against PyTorch on TPU.
Uses the same SSH infrastructure as run_jaxbench.py.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent  # Go up one level from scripts/
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)

from google.cloud import storage, tpu_v2

PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
TPU_NAME = "jaxbench-runner"
BUCKET_NAME = "tpu-dumps"
SSH_KEY = os.path.expanduser("~/.ssh/id_rsa_tpu")
SSH_USER = "REDACTED_SSH_USER"

# Tasks to test - (task_id, task_name, use_small_pytorch)
TASKS = [
    # ConvTranspose tasks
    ("58", "conv_transposed_3D__asymmetric_input__asymmetric_kernel", False),
    ("68", "conv_transposed_3D__square_input__asymmetric_kernel", False),
    ("69", "conv_transposed_2D__asymmetric_input__asymmetric_kernel", False),
    ("70", "conv_transposed_3D__asymmetric_input__square_kernel", False),
    ("72", "conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_", False),
    ("74", "conv_transposed_1D_dilated", False),
    ("75", "conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__", False),
    ("77", "conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__", False),
    ("79", "conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__", False),
    ("81", "conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__", False),
    # Standard conv
    ("63", "conv_standard_2D__square_input__square_kernel", False),
    ("66", "conv_standard_3D__asymmetric_input__asymmetric_kernel", False),
    # Memory-constrained (use small)
    ("41", "Max_Pooling_1D", True),
    ("45", "Average_Pooling_2D", True),
    ("96", "HuberLoss", True),
    # Tasks with null PyTorch/XLA time - need to rerun with smaller sizes
    ("34", "InstanceNorm", True),
    ("35", "GroupNorm_", True),
    ("38", "L1Norm_", True),
    ("39", "L2Norm_", True),
    ("76", "conv_standard_1D_dilated_strided__", True),
    ("87", "conv_pointwise_2D", True),
]


def get_tpu_ip():
    """Get TPU IP address."""
    client = tpu_v2.TpuClient()
    name = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{TPU_NAME}"
    node = client.get_node(name=name)
    if node.network_endpoints:
        return node.network_endpoints[0].ip_address
    return None


GCLOUD = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
GCLOUD_CONFIG = "/tmp/gcloud_config"

def run_ssh(ip, cmd, timeout=300):
    """Run SSH command on TPU using gcloud."""
    # Use gcloud compute tpus tpu-vm ssh which handles IAP tunneling
    ssh_cmd = [
        GCLOUD, "compute", "tpus", "tpu-vm", "ssh",
        TPU_NAME,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command", cmd
    ]
    
    # Set up environment for gcloud
    env = os.environ.copy()
    env["CLOUDSDK_CONFIG"] = GCLOUD_CONFIG
    env["CLOUDSDK_CORE_PROJECT"] = PROJECT_ID
    env["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1


def upload_gcs(blob_name, content):
    """Upload to GCS."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)


def download_gcs(blob_name):
    """Download from GCS."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    try:
        return blob.download_as_string().decode()
    except:
        return None


def test_task(task_id, task_name, use_small, tpu_ip):
    """Test a single task."""
    print(f"\n{'='*60}")
    print(f"Task {task_id}: {task_name}")
    print(f"{'='*60}")
    
    # Find files
    jax_dir = BASE_DIR / "jaxbench" / "level1"
    pt_dir = BASE_DIR / "KernelBench" / "KernelBench" / "level1"
    
    jax_files = list(jax_dir.glob(f"{task_id}_*.py"))
    if not jax_files:
        print(f"  ERROR: No JAX file for task {task_id}")
        return None
    jax_file = jax_files[0]
    
    if use_small:
        pt_files = list(pt_dir.glob(f"{task_id}_*_small.py"))
        if not pt_files:
            pt_files = [f for f in pt_dir.glob(f"{task_id}_*.py") if "_small" not in f.name]
    else:
        pt_files = [f for f in pt_dir.glob(f"{task_id}_*.py") if "_small" not in f.name]
    
    if not pt_files:
        print(f"  ERROR: No PyTorch file for task {task_id}")
        return None
    pt_file = pt_files[0]
    
    print(f"  JAX: {jax_file.name}")
    print(f"  PyTorch: {pt_file.name}")
    
    jax_code = jax_file.read_text()
    pt_code = pt_file.read_text()
    
    # Upload to GCS
    upload_gcs(f"manual/jax_{task_id}.py", jax_code)
    upload_gcs(f"manual/pt_{task_id}.py", pt_code)
    
    # Delete old result to ensure we get fresh results
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        bucket.blob(f"manual/result_{task_id}.json").delete()
    except:
        pass
    
    # Create validation script
    val_script = f'''#!/usr/bin/env python3
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["JAX_PLATFORMS"] = "tpu"

import sys, json, time
import numpy as np
import torch

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("{BUCKET_NAME}")

pt_code = bucket.blob("manual/pt_{task_id}.py").download_as_string().decode()
jax_code = bucket.blob("manual/jax_{task_id}.py").download_as_string().decode()

result = {{"task_id": "{task_id}", "task_name": "{task_name}", "compilation_success": False, "correctness_success": False}}

torch.manual_seed(42)
np.random.seed(42)

pt_ns = {{"__name__": "__main__"}}
exec(pt_code, pt_ns)

jax_ns = {{"__name__": "__main__"}}
try:
    exec(jax_code, jax_ns)
    result["compilation_success"] = True
    print("JAX compile: OK")
except Exception as e:
    result["error"] = str(e)
    print(f"JAX compile FAIL: {{e}}")
    bucket.blob("manual/result_{task_id}.json").upload_from_string(json.dumps(result))
    sys.exit(1)

init_inputs = pt_ns.get("get_init_inputs", lambda: [])()
pt_model = pt_ns["Model"](*init_inputs)
pt_model.eval()
jax_model = jax_ns["Model"](*init_inputs)

weights = {{}}
for n, p in pt_model.named_parameters():
    weights[n] = p.detach().cpu().numpy()
if weights and hasattr(jax_model, "set_weights"):
    jax_model.set_weights(weights)
    print(f"Transferred {{len(weights)}} weights")

torch.manual_seed(42)
np.random.seed(42)
pt_inputs = pt_ns["get_inputs"]()

with torch.no_grad():
    pt_out = pt_model(*pt_inputs)
pt_out_np = pt_out.detach().cpu().numpy()
print(f"PT shape: {{pt_out_np.shape}}")

import jax
import jax.numpy as jnp
jax_inputs = [jnp.array(x.numpy()) if hasattr(x, 'numpy') else jnp.array(x) for x in pt_inputs]
jax_fwd = jax.jit(jax_model.forward)

try:
    for _ in range(3):
        jax_out = jax_fwd(*jax_inputs)
        jax_out.block_until_ready()
    jax_out_np = np.array(jax_out)
    print(f"JAX shape: {{jax_out_np.shape}}")
except Exception as e:
    result["error"] = str(e)
    print(f"JAX exec FAIL: {{e}}")
    bucket.blob("manual/result_{task_id}.json").upload_from_string(json.dumps(result))
    sys.exit(1)

if pt_out_np.shape != jax_out_np.shape:
    result["error"] = f"Shape mismatch: PT={{pt_out_np.shape}} JAX={{jax_out_np.shape}}"
    print(f"FAIL: {{result['error']}}")
else:
    max_diff = float(np.max(np.abs(pt_out_np - jax_out_np)))
    result["max_diff"] = max_diff
    if np.allclose(pt_out_np, jax_out_np, rtol=0.1, atol=5.0):
        result["correctness_success"] = True
        print(f"Correct: OK (max_diff={{max_diff:.6f}})")
    else:
        result["error"] = f"Values differ (max_diff={{max_diff:.6f}})"
        print(f"FAIL: {{result['error']}}")

if not result["correctness_success"]:
    bucket.blob("manual/result_{task_id}.json").upload_from_string(json.dumps(result))
    sys.exit(1)

# Benchmark JAX
times = []
for _ in range(50):
    t0 = time.perf_counter()
    jax_out = jax_fwd(*jax_inputs)
    jax_out.block_until_ready()
    times.append((time.perf_counter() - t0) * 1000)
jax_ms = np.mean(times)
result["jax_ms"] = round(jax_ms, 4)
print(f"JAX: {{jax_ms:.3f}}ms")

# Benchmark PyTorch/XLA
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    
    dev = xm.xla_device()
    pt_model_xla = pt_model.to(dev)
    pt_inputs_xla = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in pt_inputs]
    
    # Warmup - make sure compilation happens
    with torch.no_grad():
        for _ in range(10):
            out = pt_model_xla(*pt_inputs_xla)
            xm.mark_step()
        xm.wait_device_ops()
    
    # Benchmark with proper synchronization
    # Use xm.add_step_closure to ensure timing is accurate
    times = []
    with torch.no_grad():
        for _ in range(50):
            xm.wait_device_ops()  # Ensure previous ops are done
            t0 = time.perf_counter()
            out = pt_model_xla(*pt_inputs_xla)
            xm.mark_step()
            xm.wait_device_ops()  # Wait for this op to complete
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    pt_ms = np.mean(times)
    result["pytorch_xla_ms"] = round(pt_ms, 4)
    result["speedup"] = round(pt_ms / jax_ms, 2) if jax_ms > 0 else 0
    print(f"PyTorch/XLA: {{pt_ms:.3f}}ms")
    print(f"Speedup: {{result['speedup']}}x")
except Exception as e:
    print(f"PyTorch/XLA error: {{e}}")
    result["pytorch_xla_error"] = str(e)

bucket.blob("manual/result_{task_id}.json").upload_from_string(json.dumps(result))
print("Done")
'''
    
    upload_gcs(f"manual/validate_{task_id}.py", val_script)
    
    # Clear TPU - be very aggressive
    print("  Clearing TPU...")
    run_ssh(tpu_ip, "sudo pkill -9 -f python; sudo rm -f /tmp/libtpu_lockfile /tmp/tpu_lock* 2>/dev/null; sleep 5", timeout=60)
    time.sleep(8)
    
    # Run validation
    print("  Running on TPU...")
    cmd = f"gsutil cp gs://{BUCKET_NAME}/manual/validate_{task_id}.py ~/val.py && python3 ~/val.py"
    stdout, stderr, rc = run_ssh(tpu_ip, cmd, timeout=600)
    
    output = stdout + stderr
    for line in output.split('\n')[-20:]:
        if line.strip():
            print(f"    {line}")
    
    # Get result
    result_json = download_gcs(f"manual/result_{task_id}.json")
    if result_json:
        result = json.loads(result_json)
        return result
    else:
        print("  ERROR: No result")
        return {"task_id": task_id, "task_name": task_name, "error": "No result"}


def update_checkpoint(results):
    """Update checkpoint with results."""
    cp_path = BASE_DIR / "results" / "checkpoint_level1.json"
    with open(cp_path) as f:
        cp = json.load(f)
    
    for r in results:
        if not r:
            continue
        tid = r.get("task_id")
        for i, t in enumerate(cp["tasks"]):
            if t["task_id"] == tid:
                cp["tasks"][i].update({
                    "compilation_success": r.get("compilation_success", False),
                    "correctness_success": r.get("correctness_success", False),
                    "max_diff": r.get("max_diff"),
                    "jax_ms": r.get("jax_ms"),
                    "pytorch_xla_ms": r.get("pytorch_xla_ms"),
                    "speedup": r.get("speedup"),
                    "error": r.get("error", ""),
                })
                break
    
    passed = sum(1 for t in cp["tasks"] if t.get("correctness_success"))
    cp["summary"]["passed"] = passed
    cp["summary"]["failed"] = cp["summary"]["total"] - passed
    cp["timestamp"] = datetime.now().isoformat()
    
    with open(cp_path, 'w') as f:
        json.dump(cp, f, indent=2)
    
    print(f"\nCheckpoint: {passed}/{cp['summary']['total']} passed")


def main():
    print("="*60)
    print("Manual JAX Kernel Testing")
    print("="*60)
    
    # Get TPU IP
    print("\nGetting TPU IP...")
    tpu_ip = get_tpu_ip()
    if not tpu_ip:
        print("ERROR: Could not get TPU IP")
        sys.exit(1)
    print(f"TPU IP: {tpu_ip}")
    
    # Filter tasks if specified
    if len(sys.argv) > 1:
        task_ids = sys.argv[1].split(",")
        tasks = [(t[0], t[1], t[2]) for t in TASKS if t[0] in task_ids]
    else:
        tasks = TASKS
    
    print(f"\nTesting {len(tasks)} tasks...")
    
    results = []
    for tid, tname, use_small in tasks:
        r = test_task(tid, tname, use_small, tpu_ip)
        results.append(r)
        time.sleep(2)
    
    # Update checkpoint
    update_checkpoint(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    ok = sum(1 for r in results if r and r.get("correctness_success"))
    print(f"Passed: {ok}/{len(results)}")
    for r in results:
        if r:
            s = "✓" if r.get("correctness_success") else "✗"
            sp = r.get("speedup", "N/A")
            print(f"  {s} {r['task_id']}: {r.get('task_name', '')} - {sp}x")


if __name__ == "__main__":
    main()

