#!/usr/bin/env python3
"""
Manual testing script for individual JAX implementations against PyTorch.
This runs on the TPU and validates correctness + performance.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from google.cloud import storage
from google.cloud import tpu_v2

# Configuration
PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
TPU_NAME = "jaxbench-runner"
BUCKET_NAME = "tpu-dumps"

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(__file__).parent / "credentials.json")


def get_tpu_ip():
    """Get the TPU internal IP address."""
    client = tpu_v2.TpuClient()
    name = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{TPU_NAME}"
    try:
        node = client.get_node(name=name)
        if node.network_endpoints:
            return node.network_endpoints[0].ip_address
        return None
    except Exception as e:
        print(f"Error getting TPU: {e}")
        return None


def run_ssh_command(ip, cmd, timeout=300):
    """Run a command on the TPU via SSH."""
    ssh_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh",
        TPU_NAME,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command", cmd
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1


def upload_to_gcs(bucket_name, blob_name, content):
    """Upload content to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)


def download_from_gcs(bucket_name, blob_name):
    """Download content from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        return blob.download_as_string().decode()
    except:
        return None


def test_kernel(task_id, pytorch_code, jax_code, rtol=0.1, atol=5.0):
    """Test a JAX kernel against PyTorch on TPU."""
    print(f"\n{'='*60}")
    print(f"Testing Task {task_id}")
    print(f"{'='*60}")
    
    # Upload code to GCS
    upload_to_gcs(BUCKET_NAME, f"test/pytorch_{task_id}.py", pytorch_code)
    upload_to_gcs(BUCKET_NAME, f"test/jax_{task_id}.py", jax_code)
    
    # Create validation script
    validation_script = f'''#!/usr/bin/env python3
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["JAX_PLATFORMS"] = "tpu"

import sys
import json
import time
import numpy as np

# Download code from GCS
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("{BUCKET_NAME}")

pt_code = bucket.blob("test/pytorch_{task_id}.py").download_as_string().decode()
jax_code = bucket.blob("test/jax_{task_id}.py").download_as_string().decode()

result = {{"task_id": "{task_id}", "compilation_success": False, "correctness_success": False}}

# Execute PyTorch code
pt_ns = {{"__name__": "__main__"}}
exec(pt_code, pt_ns)

# Execute JAX code  
jax_ns = {{"__name__": "__main__"}}
try:
    exec(jax_code, jax_ns)
    result["compilation_success"] = True
    print("  JAX compilation: OK")
except Exception as e:
    result["error"] = str(e)
    print(f"  JAX compilation FAILED: {{e}}")
    bucket.blob("test/result_{task_id}.json").upload_from_string(json.dumps(result))
    sys.exit(1)

# Get init inputs if available
init_inputs = pt_ns.get("get_init_inputs", lambda: [])()

# Create models
pt_model = pt_ns["Model"](*init_inputs)
pt_model.eval()

jax_model = jax_ns["Model"](*init_inputs)

# Transfer weights from PyTorch to JAX
weights_dict = {{}}
for name, param in pt_model.named_parameters():
    weights_dict[name] = param.detach().cpu().numpy()

if weights_dict and hasattr(jax_model, "set_weights"):
    jax_model.set_weights(weights_dict)
    print(f"  Transferred {{len(weights_dict)}} weights from PyTorch to JAX")

# Get inputs with fixed seed
np.random.seed(42)
import torch
torch.manual_seed(42)
pt_inputs = pt_ns["get_inputs"]()

# Run PyTorch reference
with torch.no_grad():
    pt_out = pt_model(*pt_inputs)
pt_out_np = pt_out.detach().cpu().numpy()
print(f"  PT output shape: {{pt_out_np.shape}}")

# Run JAX
import jax
import jax.numpy as jnp

jax_inputs = [jnp.array(x.numpy()) if hasattr(x, 'numpy') else jnp.array(x) for x in pt_inputs]
jax_forward = jax.jit(jax_model.forward)

# Warmup
for _ in range(3):
    jax_out = jax_forward(*jax_inputs)
    jax_out.block_until_ready()

jax_out_np = np.array(jax_out)
print(f"  JAX output shape: {{jax_out_np.shape}}")

# Check correctness
if pt_out_np.shape != jax_out_np.shape:
    result["error"] = f"Shape mismatch: PT={{pt_out_np.shape}} vs JAX={{jax_out_np.shape}}"
    print(f"  FAIL: {{result['error']}}")
else:
    max_diff = float(np.max(np.abs(pt_out_np - jax_out_np)))
    result["max_diff"] = max_diff
    
    if np.allclose(pt_out_np, jax_out_np, rtol={rtol}, atol={atol}):
        result["correctness_success"] = True
        print(f"  Correctness: OK (max_diff={{max_diff:.6f}})")
    else:
        result["error"] = f"Values differ (max_diff={{max_diff:.6f}})"
        print(f"  FAIL: {{result['error']}}")

if not result["correctness_success"]:
    bucket.blob("test/result_{task_id}.json").upload_from_string(json.dumps(result))
    sys.exit(1)

# Benchmark JAX
times = []
for _ in range(50):
    t0 = time.perf_counter()
    jax_out = jax_forward(*jax_inputs)
    jax_out.block_until_ready()
    times.append((time.perf_counter() - t0) * 1000)
jax_ms = np.mean(times)
result["jax_ms"] = round(jax_ms, 4)
print(f"  JAX TPU: {{jax_ms:.3f}}ms")

# Benchmark PyTorch/XLA
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    
    dev = xm.xla_device()
    pt_model_xla = pt_model.to(dev)
    pt_inputs_xla = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in pt_inputs]
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            pt_model_xla(*pt_inputs_xla)
            xm.mark_step()
            xm.wait_device_ops()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            pt_model_xla(*pt_inputs_xla)
            xm.mark_step()
            xm.wait_device_ops()
            times.append((time.perf_counter() - t0) * 1000)
    
    pt_xla_ms = np.mean(times)
    result["pytorch_xla_ms"] = round(pt_xla_ms, 4)
    result["speedup"] = round(pt_xla_ms / jax_ms, 2) if jax_ms > 0 else 0
    print(f"  PyTorch/XLA TPU: {{pt_xla_ms:.3f}}ms")
    print(f"  Speedup: {{result['speedup']}}x")
except Exception as e:
    print(f"  PyTorch/XLA error: {{e}}")
    result["pytorch_xla_error"] = str(e)

# Save result
bucket.blob("test/result_{task_id}.json").upload_from_string(json.dumps(result))
print("  Result saved to GCS")
'''
    
    upload_to_gcs(BUCKET_NAME, f"test/validate_{task_id}.py", validation_script)
    
    # Clear TPU state
    print("  Clearing TPU state...")
    run_ssh_command(get_tpu_ip(), "sudo pkill -9 python; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true", timeout=30)
    time.sleep(2)
    
    # Run validation
    print("  Running validation on TPU...")
    cmd = f'''
cd ~ && \
gsutil cp gs://{BUCKET_NAME}/test/validate_{task_id}.py validate.py && \
python3 validate.py 2>&1
'''
    stdout, stderr, rc = run_ssh_command(get_tpu_ip(), cmd, timeout=300)
    
    print("  TPU Output:")
    for line in (stdout + stderr).split('\n'):
        if line.strip():
            print(f"    {line}")
    
    # Get result from GCS
    result_json = download_from_gcs(BUCKET_NAME, f"test/result_{task_id}.json")
    if result_json:
        result = json.loads(result_json)
        print(f"\n  Final Result: {json.dumps(result, indent=2)}")
        return result
    else:
        print("  ERROR: Could not retrieve result from GCS")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_test.py <task_id>")
        print("  This will test the JAX code in jaxbench/level1/<task_id>_*.py")
        sys.exit(1)
    
    task_id = sys.argv[1]
    
    # Find the files
    jaxbench_dir = Path(__file__).parent / "jaxbench" / "level1"
    kernelbench_dir = Path(__file__).parent / "KernelBench" / "KernelBench" / "level1"
    
    # Find JAX file
    jax_files = list(jaxbench_dir.glob(f"{task_id}_*.py"))
    if not jax_files:
        print(f"No JAX file found for task {task_id}")
        sys.exit(1)
    jax_file = jax_files[0]
    
    # Find PyTorch file
    pt_files = list(kernelbench_dir.glob(f"{task_id}_*.py"))
    if not pt_files:
        print(f"No PyTorch file found for task {task_id}")
        sys.exit(1)
    pt_file = pt_files[0]
    
    print(f"JAX file: {jax_file}")
    print(f"PyTorch file: {pt_file}")
    
    with open(jax_file) as f:
        jax_code = f.read()
    with open(pt_file) as f:
        pytorch_code = f.read()
    
    result = test_kernel(task_id, pytorch_code, jax_code)

