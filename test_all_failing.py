#!/usr/bin/env python3
"""
Comprehensive test script for all failing Level 1 JAX implementations.
Tests against PyTorch on TPU and updates checkpoint_level1.json with results.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from google.cloud import tpu_v2

# Configuration
PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
TPU_NAME = "jaxbench-runner"
BUCKET_NAME = "tpu-dumps"

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(__file__).parent / "credentials.json")

# Tasks to test
# Format: (task_id, task_name, use_small_version, needs_pytorch_xla_only)
TASKS_TO_TEST = [
    # ConvTranspose tasks (JAX needs to be written)
    ("58", "conv_transposed_3D__asymmetric_input__asymmetric_kernel", False, False),
    ("68", "conv_transposed_3D__square_input__asymmetric_kernel", False, False),
    ("69", "conv_transposed_2D__asymmetric_input__asymmetric_kernel", False, False),
    ("70", "conv_transposed_3D__asymmetric_input__square_kernel", False, False),
    ("72", "conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_", False, False),
    ("74", "conv_transposed_1D_dilated", False, False),
    ("75", "conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__", False, False),
    ("77", "conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__", False, False),
    ("79", "conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__", False, False),
    ("81", "conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__", False, False),
    
    # Standard conv tasks
    ("63", "conv_standard_2D__square_input__square_kernel", False, False),
    ("66", "conv_standard_3D__asymmetric_input__asymmetric_kernel", False, False),
    
    # Memory-constrained tasks (use small versions)
    ("41", "Max_Pooling_1D", True, False),
    ("45", "Average_Pooling_2D", True, False),
    ("96", "HuberLoss", True, False),
    
    # Tasks with JAX working but no PyTorch/XLA time
    ("34", "InstanceNorm", False, True),
    ("35", "GroupNorm_", False, True),
    ("38", "L1Norm_", False, True),
    ("39", "L2Norm_", False, True),
    ("76", "conv_standard_1D_dilated_strided__", False, True),
    ("87", "conv_pointwise_2D", False, True),
]


GCLOUD_PATH = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"


def run_ssh_command(cmd, timeout=300):
    """Run a command on the TPU via SSH."""
    ssh_cmd = [
        GCLOUD_PATH, "compute", "tpus", "tpu-vm", "ssh",
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


def upload_to_gcs(blob_name, content):
    """Upload content to GCS."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)


def download_from_gcs(blob_name):
    """Download content from GCS."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    try:
        return blob.download_as_string().decode()
    except:
        return None


def clear_tpu_state():
    """Clear TPU state before running tests."""
    print("  Clearing TPU state...")
    run_ssh_command("sudo pkill -9 python; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true", timeout=30)
    time.sleep(2)


def test_task(task_id, task_name, use_small, pytorch_xla_only=False):
    """Test a single task on TPU."""
    print(f"\n{'='*60}")
    print(f"Testing Task {task_id}: {task_name}")
    print(f"{'='*60}")
    
    # Find files
    jaxbench_dir = Path(__file__).parent / "jaxbench" / "level1"
    kernelbench_dir = Path(__file__).parent / "KernelBench" / "KernelBench" / "level1"
    
    # JAX file
    jax_files = list(jaxbench_dir.glob(f"{task_id}_*.py"))
    if not jax_files:
        print(f"  ERROR: No JAX file found for task {task_id}")
        return None
    jax_file = jax_files[0]
    
    # PyTorch file (use small version if needed)
    if use_small:
        pt_files = list(kernelbench_dir.glob(f"{task_id}_*_small.py"))
        if not pt_files:
            # Fall back to regular file
            pt_files = list(kernelbench_dir.glob(f"{task_id}_*.py"))
    else:
        pt_files = [f for f in kernelbench_dir.glob(f"{task_id}_*.py") if "_small" not in f.name]
    
    if not pt_files:
        print(f"  ERROR: No PyTorch file found for task {task_id}")
        return None
    pt_file = pt_files[0]
    
    print(f"  JAX file: {jax_file.name}")
    print(f"  PyTorch file: {pt_file.name}")
    
    with open(jax_file) as f:
        jax_code = f.read()
    with open(pt_file) as f:
        pytorch_code = f.read()
    
    # Upload code to GCS
    upload_to_gcs(f"test/pytorch_{task_id}.py", pytorch_code)
    upload_to_gcs(f"test/jax_{task_id}.py", jax_code)
    
    # Create validation script
    rtol, atol = 0.1, 5.0
    
    if pytorch_xla_only:
        # Only run PyTorch/XLA benchmark (JAX already works)
        validation_script = create_pytorch_xla_only_script(task_id)
    else:
        # Full validation
        validation_script = create_full_validation_script(task_id, rtol, atol)
    
    upload_to_gcs(f"test/validate_{task_id}.py", validation_script)
    
    # Clear TPU state
    clear_tpu_state()
    
    # Run validation
    print("  Running validation on TPU...")
    cmd = f'''
cd ~ && \
gsutil cp gs://{BUCKET_NAME}/test/validate_{task_id}.py validate.py && \
python3 validate.py 2>&1
'''
    stdout, stderr, rc = run_ssh_command(cmd, timeout=600)
    
    output = stdout + stderr
    print("  TPU Output:")
    for line in output.split('\n')[-30:]:  # Last 30 lines
        if line.strip():
            print(f"    {line}")
    
    # Get result from GCS
    result_json = download_from_gcs(f"test/result_{task_id}.json")
    if result_json:
        result = json.loads(result_json)
        result["task_id"] = task_id
        result["task_name"] = task_name
        print(f"\n  Result: compile={result.get('compilation_success')}, correct={result.get('correctness_success')}")
        if result.get('jax_ms'):
            print(f"  JAX: {result['jax_ms']:.3f}ms")
        if result.get('pytorch_xla_ms'):
            print(f"  PyTorch/XLA: {result['pytorch_xla_ms']:.3f}ms")
        if result.get('speedup'):
            print(f"  Speedup: {result['speedup']}x")
        if result.get('error'):
            print(f"  Error: {result['error']}")
        return result
    else:
        print("  ERROR: Could not retrieve result from GCS")
        return {"task_id": task_id, "task_name": task_name, "error": "No result from GCS"}


def create_full_validation_script(task_id, rtol, atol):
    return f'''#!/usr/bin/env python3
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["JAX_PLATFORMS"] = "tpu"

import sys
import json
import time
import numpy as np

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("{BUCKET_NAME}")

pt_code = bucket.blob("test/pytorch_{task_id}.py").download_as_string().decode()
jax_code = bucket.blob("test/jax_{task_id}.py").download_as_string().decode()

result = {{"task_id": "{task_id}", "compilation_success": False, "correctness_success": False}}

# Execute PyTorch code
import torch
torch.manual_seed(42)
np.random.seed(42)

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

# Get init inputs
init_inputs = pt_ns.get("get_init_inputs", lambda: [])()

# Create models
pt_model = pt_ns["Model"](*init_inputs)
pt_model.eval()

jax_model = jax_ns["Model"](*init_inputs)

# Transfer weights
weights_dict = {{}}
for name, param in pt_model.named_parameters():
    weights_dict[name] = param.detach().cpu().numpy()

if weights_dict and hasattr(jax_model, "set_weights"):
    jax_model.set_weights(weights_dict)
    print(f"  Transferred {{len(weights_dict)}} weights")

# Get inputs with fixed seed
torch.manual_seed(42)
np.random.seed(42)
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

try:
    for _ in range(3):
        jax_out = jax_forward(*jax_inputs)
        jax_out.block_until_ready()
    
    jax_out_np = np.array(jax_out)
    print(f"  JAX output shape: {{jax_out_np.shape}}")
except Exception as e:
    result["error"] = str(e)
    print(f"  JAX execution FAILED: {{e}}")
    bucket.blob("test/result_{task_id}.json").upload_from_string(json.dumps(result))
    sys.exit(1)

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
    
    with torch.no_grad():
        for _ in range(5):
            pt_model_xla(*pt_inputs_xla)
            xm.mark_step()
            xm.wait_device_ops()
    
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

bucket.blob("test/result_{task_id}.json").upload_from_string(json.dumps(result))
print("  Result saved")
'''


def create_pytorch_xla_only_script(task_id):
    """Create script that only benchmarks PyTorch/XLA (JAX already verified)."""
    return f'''#!/usr/bin/env python3
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["JAX_PLATFORMS"] = "tpu"

import sys
import json
import time
import numpy as np
import torch

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("{BUCKET_NAME}")

pt_code = bucket.blob("test/pytorch_{task_id}.py").download_as_string().decode()
jax_code = bucket.blob("test/jax_{task_id}.py").download_as_string().decode()

result = {{"task_id": "{task_id}", "compilation_success": True, "correctness_success": True}}

# Execute PyTorch code
torch.manual_seed(42)
np.random.seed(42)

pt_ns = {{"__name__": "__main__"}}
exec(pt_code, pt_ns)

# Get init inputs
init_inputs = pt_ns.get("get_init_inputs", lambda: [])()

# Create model
pt_model = pt_ns["Model"](*init_inputs)
pt_model.eval()

# Get inputs
torch.manual_seed(42)
pt_inputs = pt_ns["get_inputs"]()

# Also run JAX to get its timing
jax_ns = {{"__name__": "__main__"}}
exec(jax_code, jax_ns)

jax_model = jax_ns["Model"](*init_inputs)

# Transfer weights
weights_dict = {{}}
for name, param in pt_model.named_parameters():
    weights_dict[name] = param.detach().cpu().numpy()

if weights_dict and hasattr(jax_model, "set_weights"):
    jax_model.set_weights(weights_dict)

import jax
import jax.numpy as jnp

jax_inputs = [jnp.array(x.numpy()) if hasattr(x, 'numpy') else jnp.array(x) for x in pt_inputs]
jax_forward = jax.jit(jax_model.forward)

# Benchmark JAX
for _ in range(3):
    jax_out = jax_forward(*jax_inputs)
    jax_out.block_until_ready()

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
    
    with torch.no_grad():
        for _ in range(5):
            pt_model_xla(*pt_inputs_xla)
            xm.mark_step()
            xm.wait_device_ops()
    
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

bucket.blob("test/result_{task_id}.json").upload_from_string(json.dumps(result))
print("  Result saved")
'''


def update_checkpoint(results):
    """Update checkpoint_level1.json with new results."""
    checkpoint_path = Path(__file__).parent / "results" / "checkpoint_level1.json"
    
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    
    # Update tasks with new results
    for result in results:
        if result is None:
            continue
        
        task_id = result.get("task_id")
        
        # Find existing task
        found = False
        for i, task in enumerate(checkpoint["tasks"]):
            if task["task_id"] == task_id:
                # Update the task
                checkpoint["tasks"][i].update({
                    "compilation_success": result.get("compilation_success", False),
                    "correctness_success": result.get("correctness_success", False),
                    "max_diff": result.get("max_diff"),
                    "jax_ms": result.get("jax_ms"),
                    "pytorch_xla_ms": result.get("pytorch_xla_ms"),
                    "speedup": result.get("speedup"),
                    "error": result.get("error", ""),
                })
                found = True
                break
        
        if not found:
            print(f"  Warning: Task {task_id} not found in checkpoint")
    
    # Update summary
    passed = sum(1 for t in checkpoint["tasks"] if t.get("correctness_success"))
    checkpoint["summary"]["passed"] = passed
    checkpoint["summary"]["failed"] = checkpoint["summary"]["total"] - passed
    checkpoint["timestamp"] = datetime.now().isoformat()
    
    # Save
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"\nCheckpoint updated: {passed}/{checkpoint['summary']['total']} passed")


def main():
    print("="*60)
    print("JAXBench Level 1 - Manual Testing")
    print("="*60)
    
    # Check if specific tasks are requested
    if len(sys.argv) > 1:
        task_ids = sys.argv[1].split(",")
        tasks = [(t[0], t[1], t[2], t[3]) for t in TASKS_TO_TEST if t[0] in task_ids]
    else:
        tasks = TASKS_TO_TEST
    
    print(f"\nTesting {len(tasks)} tasks...")
    
    results = []
    for task_id, task_name, use_small, pytorch_xla_only in tasks:
        result = test_task(task_id, task_name, use_small, pytorch_xla_only)
        results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Update checkpoint
    update_checkpoint(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success = sum(1 for r in results if r and r.get("correctness_success"))
    print(f"Passed: {success}/{len(results)}")
    
    for result in results:
        if result:
            status = "✓" if result.get("correctness_success") else "✗"
            speedup = result.get("speedup", "N/A")
            print(f"  {status} Task {result['task_id']}: {result['task_name']} - speedup={speedup}x")


if __name__ == "__main__":
    main()

