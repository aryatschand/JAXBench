#!/usr/bin/env python3
"""
Simple TPU benchmark: JAX vs PyTorch/XLA on TPU v6e-1
"""

import json
import time
from google.oauth2 import service_account
from google.cloud import tpu_v2, storage
from google.api_core import exceptions

PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
BUCKET_NAME = "tpu-dumps"
TPU_NAME = "bench-simple"

credentials = service_account.Credentials.from_service_account_file(
    "credentials.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

tpu_client = tpu_v2.TpuClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

# Very simple startup script
STARTUP_SCRIPT = r'''#!/bin/bash
set -ex

# Log everything
exec > /tmp/bench.log 2>&1
echo "=== Starting benchmark at $(date) ==="

# Install packages
pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -q torch numpy google-cloud-storage

# Try to install torch_xla (may fail)
pip install -q torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html || echo "torch_xla install failed"

# Run benchmark
python3 -c '
import json
import time
import numpy as np

print("=== BENCHMARK ===")
results = {}

# JAX
print("\n--- JAX ---")
import jax
import jax.numpy as jnp
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

N = 4096
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N))
B = jax.random.uniform(key, (N, N))

@jax.jit
def mm(a, b):
    return jnp.matmul(a, b)

# warmup
for _ in range(5):
    mm(A, B).block_until_ready()

# benchmark
times = []
for _ in range(20):
    t0 = time.perf_counter()
    mm(A, B).block_until_ready()
    times.append((time.perf_counter() - t0) * 1000)

results["jax_tpu_ms"] = round(np.mean(times), 3)
print(f"JAX TPU: {results['jax_tpu_ms']} ms")

# PyTorch CPU
print("\n--- PyTorch CPU ---")
import torch
A_pt = torch.rand(N, N)
B_pt = torch.rand(N, N)

with torch.no_grad():
    for _ in range(3):
        torch.matmul(A_pt, B_pt)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        torch.matmul(A_pt, B_pt)
        times.append((time.perf_counter() - t0) * 1000)

results["pytorch_cpu_ms"] = round(np.mean(times), 3)
print(f"PyTorch CPU: {results['pytorch_cpu_ms']} ms")

# PyTorch/XLA
print("\n--- PyTorch/XLA ---")
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print(f"Device: {dev}")
    
    A_xla = torch.rand(N, N, device=dev)
    B_xla = torch.rand(N, N, device=dev)
    
    with torch.no_grad():
        for _ in range(5):
            torch.matmul(A_xla, B_xla)
            xm.mark_step()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            torch.matmul(A_xla, B_xla)
            xm.mark_step()
            times.append((time.perf_counter() - t0) * 1000)
    
    results["pytorch_xla_tpu_ms"] = round(np.mean(times), 3)
    print(f"PyTorch/XLA TPU: {results['pytorch_xla_tpu_ms']} ms")
except Exception as e:
    results["pytorch_xla_error"] = str(e)[:200]
    print(f"PyTorch/XLA error: {e}")

# Summary
print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))

# Upload
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("tpu-dumps")
bucket.blob("simple_bench.json").upload_from_string(json.dumps(results))
print("Uploaded!")
'

echo "=== Done at $(date) ==="
'''

def main():
    print("=" * 60)
    print("Simple TPU Benchmark: JAX vs PyTorch/XLA")
    print("=" * 60)
    
    # Clear old results
    print("\n[1] Clearing old results...")
    try:
        bucket.blob("simple_bench.json").delete()
        print("    Deleted old results")
    except:
        print("    No old results")
    
    # Delete any existing TPU
    print("\n[2] Cleaning up existing TPUs...")
    parent = f"projects/{PROJECT_ID}/locations/{ZONE}"
    node_path = f"{parent}/nodes/{TPU_NAME}"
    
    try:
        node = tpu_client.get_node(name=node_path)
        print(f"    Found TPU: {node.state.name}")
        if node.state.name != "DELETING":
            print("    Deleting...")
            tpu_client.delete_node(name=node_path).result(timeout=180)
            print("    Deleted, waiting 10s...")
            time.sleep(10)
    except exceptions.NotFound:
        print("    No existing TPU")
    
    # Create TPU
    print("\n[3] Creating TPU v6e-1...")
    node = tpu_v2.Node()
    node.accelerator_type = "v6e-1"
    node.runtime_version = "v2-alpha-tpuv6e"
    node.scheduling_config = tpu_v2.SchedulingConfig()
    node.scheduling_config.preemptible = True
    node.network_config = tpu_v2.NetworkConfig()
    node.network_config.network = "default"
    node.network_config.subnetwork = "default"
    node.network_config.enable_external_ips = True
    node.metadata = {"startup-script": STARTUP_SCRIPT}
    
    t0 = time.time()
    op = tpu_client.create_node(parent=parent, node_id=TPU_NAME, node=node)
    print("    Waiting for TPU to be ready...")
    
    try:
        result = op.result(timeout=300)
        print(f"    TPU ready in {time.time() - t0:.0f}s")
        if result.network_endpoints:
            ip = result.network_endpoints[0].access_config.external_ip
            print(f"    IP: {ip}")
    except Exception as e:
        print(f"    Error: {e}")
        return
    
    # Wait for benchmark results
    print("\n[4] Waiting for benchmark to complete...")
    result_blob = bucket.blob("simple_bench.json")
    
    for i in range(120):  # 10 minutes max
        time.sleep(5)
        elapsed = (i + 1) * 5
        
        if result_blob.exists():
            print(f"\n    Results found after {elapsed}s!")
            content = result_blob.download_as_string().decode()
            results = json.loads(content)
            
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(json.dumps(results, indent=2))
            
            # Analysis
            jax_ms = results.get("jax_tpu_ms")
            pt_cpu_ms = results.get("pytorch_cpu_ms")
            pt_xla_ms = results.get("pytorch_xla_tpu_ms")
            
            print("\n" + "=" * 60)
            print("ANALYSIS")
            print("=" * 60)
            if jax_ms:
                print(f"JAX on TPU:        {jax_ms:.3f} ms")
            if pt_cpu_ms:
                print(f"PyTorch on CPU:    {pt_cpu_ms:.3f} ms")
            if pt_xla_ms:
                print(f"PyTorch/XLA on TPU: {pt_xla_ms:.3f} ms")
                if jax_ms:
                    ratio = pt_xla_ms / jax_ms
                    print(f"\nJAX vs PyTorch/XLA: {ratio:.2f}x {'(JAX faster)' if ratio > 1 else '(PyTorch faster)'}")
            elif "pytorch_xla_error" in results:
                print(f"PyTorch/XLA error: {results['pytorch_xla_error']}")
            
            break
        
        if elapsed % 30 == 0:
            print(f"    [{elapsed}s] Still waiting...")
    else:
        print("\n    Timeout! Benchmark didn't complete in 10 minutes")
    
    # Cleanup
    print("\n[5] Cleaning up TPU...")
    try:
        tpu_client.delete_node(name=node_path)
        print("    Deletion initiated")
    except Exception as e:
        print(f"    Cleanup error: {e}")

if __name__ == "__main__":
    main()

