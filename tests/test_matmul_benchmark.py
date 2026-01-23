#!/usr/bin/env python3
"""
Test JAX and PyTorch/XLA execution on TPU.

This test runs KernelBench Level 1 Task 1 (Square Matrix Multiplication) on TPU
using both JAX and PyTorch/XLA to verify both frameworks work correctly.

Usage:
    python tests/test_matmul_benchmark.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set credentials
BASE_DIR = Path(__file__).parent.parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)

from google.cloud import storage, tpu_v2

# Configuration
PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
TPU_NAME = "jaxbench-runner"
BUCKET_NAME = "tpu-dumps"

# gcloud path
GCLOUD_PATHS = [
    "/opt/homebrew/share/google-cloud-sdk/bin/gcloud",
    "/usr/local/bin/gcloud",
    "gcloud"
]


def find_gcloud():
    """Find gcloud CLI."""
    for path in GCLOUD_PATHS:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return path
        except:
            continue
    return None


def get_tpu_ip():
    """Get TPU internal IP."""
    client = tpu_v2.TpuClient()
    name = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{TPU_NAME}"
    try:
        node = client.get_node(name=name)
        if node.network_endpoints:
            return node.network_endpoints[0].ip_address
    except:
        pass
    return None


def run_ssh(gcloud, cmd, timeout=300):
    """Run SSH command on TPU."""
    ssh_cmd = [
        gcloud, "compute", "tpus", "tpu-vm", "ssh",
        TPU_NAME,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command", cmd
    ]
    
    env = os.environ.copy()
    env["CLOUDSDK_CONFIG"] = "/tmp/gcloud_config"
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


def clear_tpu(gcloud):
    """Clear TPU state."""
    run_ssh(gcloud, "sudo pkill -9 -f python; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null; sleep 3", timeout=30)
    time.sleep(5)


def test_matmul():
    """Test matrix multiplication on TPU with both JAX and PyTorch/XLA."""
    print("=" * 60)
    print("JAXBench Matmul Benchmark Test")
    print("=" * 60)
    
    # Find gcloud
    gcloud = find_gcloud()
    if not gcloud:
        print("❌ gcloud CLI not found")
        return False
    print(f"✅ Found gcloud: {gcloud}")
    
    # Check TPU
    tpu_ip = get_tpu_ip()
    if not tpu_ip:
        print("❌ TPU not available")
        print("   Run: python run_jaxbench.py --keep-tpu to create one")
        return False
    print(f"✅ TPU available: {tpu_ip}")
    
    # Read KernelBench task 1
    pt_file = BASE_DIR / "KernelBench" / "KernelBench" / "level1" / "1_Square_matrix_multiplication_.py"
    jax_file = BASE_DIR / "jaxbench" / "level1" / "1_Square_matrix_multiplication_.py"
    
    if not pt_file.exists():
        print(f"❌ PyTorch file not found: {pt_file}")
        return False
    
    if not jax_file.exists():
        print(f"❌ JAX file not found: {jax_file}")
        return False
    
    print(f"✅ Found benchmark files")
    
    pt_code = pt_file.read_text()
    jax_code = jax_file.read_text()
    
    # Upload to GCS
    print("\nUploading benchmark code to GCS...")
    upload_gcs("test/matmul_pt.py", pt_code)
    upload_gcs("test/matmul_jax.py", jax_code)
    
    # Delete old result
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        bucket.blob("test/matmul_result.json").delete()
    except:
        pass
    
    # Create test script
    test_script = f'''#!/usr/bin/env python3
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["JAX_PLATFORMS"] = "tpu"

import sys, json, time
import numpy as np
import torch

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("{BUCKET_NAME}")

pt_code = bucket.blob("test/matmul_pt.py").download_as_string().decode()
jax_code = bucket.blob("test/matmul_jax.py").download_as_string().decode()

result = {{"test": "matmul", "jax_ok": False, "pytorch_xla_ok": False}}

# Test JAX
print("=" * 50)
print("Testing JAX on TPU")
print("=" * 50)

try:
    import jax
    import jax.numpy as jnp
    
    print(f"JAX version: {{jax.__version__}}")
    print(f"Backend: {{jax.default_backend()}}")
    print(f"Devices: {{jax.devices()}}")
    
    if jax.default_backend() != "tpu":
        print("WARNING: JAX not using TPU backend!")
    
    # Execute JAX code
    jax_ns = {{"__name__": "__main__"}}
    exec(jax_code, jax_ns)
    
    jax_model = jax_ns["Model"]()
    jax_inputs = jax_ns["get_inputs"]()
    
    # Convert inputs
    jax_inputs = [jnp.array(x.numpy()) if hasattr(x, 'numpy') else jnp.array(x) for x in jax_inputs]
    
    # JIT compile and warmup
    jax_fwd = jax.jit(jax_model.forward)
    for _ in range(5):
        out = jax_fwd(*jax_inputs)
        out.block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        out = jax_fwd(*jax_inputs)
        out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    
    jax_ms = np.mean(times)
    result["jax_ms"] = round(jax_ms, 4)
    result["jax_ok"] = True
    print(f"JAX TPU: {{jax_ms:.3f}} ms")
    print("✅ JAX test PASSED")
    
except Exception as e:
    print(f"❌ JAX test FAILED: {{e}}")
    result["jax_error"] = str(e)

# Test PyTorch/XLA
print()
print("=" * 50)
print("Testing PyTorch/XLA on TPU")
print("=" * 50)

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    
    print(f"torch_xla version: {{torch_xla.__version__}}")
    dev = xm.xla_device()
    print(f"XLA device: {{dev}}")
    
    # Execute PyTorch code
    pt_ns = {{"__name__": "__main__"}}
    exec(pt_code, pt_ns)
    
    pt_model = pt_ns["Model"]()
    pt_model.eval()
    pt_inputs = pt_ns["get_inputs"]()
    
    # Move to XLA device
    pt_model = pt_model.to(dev)
    pt_inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in pt_inputs]
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            out = pt_model(*pt_inputs)
            xm.mark_step()
        xm.wait_device_ops()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(50):
            xm.wait_device_ops()
            t0 = time.perf_counter()
            out = pt_model(*pt_inputs)
            xm.mark_step()
            xm.wait_device_ops()
            times.append((time.perf_counter() - t0) * 1000)
    
    pt_ms = np.mean(times)
    result["pytorch_xla_ms"] = round(pt_ms, 4)
    result["pytorch_xla_ok"] = True
    print(f"PyTorch/XLA TPU: {{pt_ms:.3f}} ms")
    print("✅ PyTorch/XLA test PASSED")
    
except Exception as e:
    print(f"❌ PyTorch/XLA test FAILED: {{e}}")
    result["pytorch_xla_error"] = str(e)

# Calculate speedup
if result.get("jax_ms") and result.get("pytorch_xla_ms"):
    speedup = result["pytorch_xla_ms"] / result["jax_ms"]
    result["speedup"] = round(speedup, 2)
    print()
    print("=" * 50)
    print(f"Speedup (PyTorch/XLA time / JAX time): {{speedup:.2f}}x")
    print("=" * 50)

# Save result
bucket.blob("test/matmul_result.json").upload_from_string(json.dumps(result, indent=2))
print()
print("Result saved to GCS")
'''
    
    upload_gcs("test/matmul_test.py", test_script)
    
    # Clear TPU state
    print("\nClearing TPU state...")
    clear_tpu(gcloud)
    
    # Run test
    print("\nRunning benchmark on TPU...")
    cmd = f"gsutil cp gs://{BUCKET_NAME}/test/matmul_test.py ~/test.py && python3 ~/test.py"
    stdout, stderr, rc = run_ssh(gcloud, cmd, timeout=300)
    
    output = stdout + stderr
    print("\n--- TPU Output ---")
    for line in output.split('\n'):
        if line.strip():
            print(f"  {line}")
    print("--- End Output ---\n")
    
    # Get result
    result_json = download_gcs("test/matmul_result.json")
    if result_json:
        result = json.loads(result_json)
        
        print("=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        jax_ok = result.get("jax_ok", False)
        pt_ok = result.get("pytorch_xla_ok", False)
        
        print(f"  JAX:          {'✅ PASSED' if jax_ok else '❌ FAILED'}")
        if result.get("jax_ms"):
            print(f"                {result['jax_ms']:.3f} ms")
        if result.get("jax_error"):
            print(f"                Error: {result['jax_error'][:100]}")
        
        print(f"  PyTorch/XLA:  {'✅ PASSED' if pt_ok else '❌ FAILED'}")
        if result.get("pytorch_xla_ms"):
            print(f"                {result['pytorch_xla_ms']:.3f} ms")
        if result.get("pytorch_xla_error"):
            print(f"                Error: {result['pytorch_xla_error'][:100]}")
        
        if result.get("speedup"):
            print(f"\n  Speedup:      {result['speedup']:.2f}x (JAX vs PyTorch/XLA)")
        
        return jax_ok and pt_ok
    else:
        print("❌ Could not retrieve results from GCS")
        return False


def main():
    """Run matmul benchmark test."""
    success = test_matmul()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All benchmark tests PASSED!")
        print("   Both JAX and PyTorch/XLA are working correctly on TPU.")
        return 0
    else:
        print("❌ Some benchmark tests FAILED!")
        print("   Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

