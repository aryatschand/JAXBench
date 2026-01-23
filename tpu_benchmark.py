"""
TPU Benchmark Script - Run JAX benchmarks on GCP TPU VMs.

This script:
1. Connects to the TPU VM
2. Installs JAX with TPU support
3. Runs benchmarks
4. Returns performance metrics
"""

import os
import subprocess
import json
import time
from google.oauth2 import service_account
from google.cloud import tpu_v2
from google.api_core import exceptions

# Configuration
PROJECT_ID = "jaxbench"
CREDENTIALS_FILE = "credentials.json"

def get_credentials():
    """Load GCP credentials."""
    creds_path = os.path.join(os.path.dirname(__file__), CREDENTIALS_FILE)
    return service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

def get_tpu_client():
    """Create TPU client."""
    return tpu_v2.TpuClient(credentials=get_credentials())

def find_tpu(name="jaxbench-test"):
    """Find a TPU by name across zones."""
    client = get_tpu_client()
    zones = ["us-central1-a", "us-central1-b", "us-central1-f", "us-central2-b", 
             "us-west4-a", "europe-west4-a"]
    
    for zone in zones:
        try:
            node_name = f"projects/{PROJECT_ID}/locations/{zone}/nodes/{name}"
            node = client.get_node(name=node_name)
            return node, zone
        except exceptions.NotFound:
            continue
    return None, None

def get_tpu_ip(tpu_name="jaxbench-test"):
    """Get TPU external IP."""
    node, zone = find_tpu(tpu_name)
    if node and node.network_endpoints:
        for ep in node.network_endpoints:
            if ep.access_config and ep.access_config.external_ip:
                return ep.access_config.external_ip
    return None

def run_command_on_tpu(tpu_name, zone, command, timeout=300):
    """
    Run a command on TPU VM using the TPU API.
    
    Note: This requires SSH key setup. For initial testing,
    we'll use gcloud or direct API.
    """
    # For TPU VMs, we need to use SSH or the gcloud command
    # The TPU API doesn't have a direct execute command
    
    # Alternative: Use metadata startup script or SSH
    print(f"Command to run on TPU:")
    print(f"  gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone} --project={PROJECT_ID} --command='{command}'")
    return None

def create_benchmark_script():
    """Create the JAX benchmark script to run on TPU."""
    return '''#!/usr/bin/env python3
"""JAX TPU Benchmark - Run on TPU VM"""

import json
import time

# Install JAX if needed
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    import subprocess
    subprocess.run([
        "pip", "install", "-q", "jax[tpu]", 
        "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    ], check=True)
    import jax
    import jax.numpy as jnp

def benchmark_matmul(sizes=[1024, 2048, 4096], num_iters=100):
    """Benchmark matrix multiplication."""
    results = []
    
    @jax.jit
    def matmul(a, b):
        return jnp.dot(a, b)
    
    for size in sizes:
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (size, size), dtype=jnp.float32)
        b = jax.random.normal(key, (size, size), dtype=jnp.float32)
        
        # Warmup
        c = matmul(a, b)
        c.block_until_ready()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            c = matmul(a, b)
            c.block_until_ready()
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / num_iters) * 1000
        tflops = (2 * size**3 * num_iters / elapsed) / 1e12
        
        results.append({
            "op": "matmul",
            "size": size,
            "avg_ms": round(avg_ms, 3),
            "tflops": round(tflops, 2)
        })
    
    return results

def benchmark_attention(batch_sizes=[8], seq_lens=[512, 1024, 2048], num_iters=100):
    """Benchmark attention mechanism."""
    results = []
    head_dim = 64
    num_heads = 8
    
    @jax.jit
    def attention(q, k, v):
        scale = 1.0 / jnp.sqrt(head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum('bhqk,bhkd->bhqd', weights, v)
    
    for batch in batch_sizes:
        for seq_len in seq_lens:
            key = jax.random.PRNGKey(0)
            shape = (batch, num_heads, seq_len, head_dim)
            q = jax.random.normal(key, shape, dtype=jnp.float32)
            k = jax.random.normal(key, shape, dtype=jnp.float32)
            v = jax.random.normal(key, shape, dtype=jnp.float32)
            
            # Warmup
            out = attention(q, k, v)
            out.block_until_ready()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iters):
                out = attention(q, k, v)
                out.block_until_ready()
            elapsed = time.perf_counter() - start
            
            avg_ms = (elapsed / num_iters) * 1000
            
            results.append({
                "op": "attention",
                "batch": batch,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "avg_ms": round(avg_ms, 3)
            })
    
    return results

def main():
    print("=" * 60)
    print("JAX TPU Benchmark")
    print("=" * 60)
    
    # Device info
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    
    all_results = {
        "device": {
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "devices": [str(d) for d in jax.devices()]
        },
        "benchmarks": {}
    }
    
    # Run benchmarks
    print("\\nRunning matmul benchmarks...")
    matmul_results = benchmark_matmul()
    all_results["benchmarks"]["matmul"] = matmul_results
    for r in matmul_results:
        print(f"  {r['size']}x{r['size']}: {r['avg_ms']:.3f} ms, {r['tflops']:.2f} TFLOPS")
    
    print("\\nRunning attention benchmarks...")
    attn_results = benchmark_attention()
    all_results["benchmarks"]["attention"] = attn_results
    for r in attn_results:
        print(f"  batch={r['batch']}, seq={r['seq_len']}: {r['avg_ms']:.3f} ms")
    
    print("\\n" + "=" * 60)
    print("Results JSON:")
    print(json.dumps(all_results, indent=2))
    
    return all_results

if __name__ == "__main__":
    main()
'''

def delete_tpu(name="jaxbench-test"):
    """Delete TPU VM."""
    node, zone = find_tpu(name)
    if not node:
        print(f"TPU {name} not found")
        return False
    
    client = get_tpu_client()
    node_path = f"projects/{PROJECT_ID}/locations/{zone}/nodes/{name}"
    
    print(f"Deleting TPU {name} in {zone}...")
    try:
        operation = client.delete_node(name=node_path)
        operation.result(timeout=300)
        print("✅ TPU deleted successfully")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_tpu(name="jaxbench-test", tpu_type="v6e-1", zone="us-central1-b", 
               preemptible=True):
    """Create a TPU VM."""
    client = get_tpu_client()
    parent = f"projects/{PROJECT_ID}/locations/{zone}"
    
    # Runtime versions for different TPU types
    runtimes = {
        "v6e": "v2-alpha-tpuv6e",
        "v5litepod": "v2-alpha-tpuv5-lite", 
        "v5p": "v2-alpha-tpuv5",
        "v2": "tpu-ubuntu2204-base",
        "v3": "tpu-ubuntu2204-base",
        "v4": "tpu-ubuntu2204-base",
    }
    
    # Find the right runtime
    runtime = "tpu-ubuntu2204-base"
    for prefix, rt in runtimes.items():
        if tpu_type.startswith(prefix):
            runtime = rt
            break
    
    node = tpu_v2.Node()
    node.accelerator_type = tpu_type
    node.runtime_version = runtime
    
    if preemptible:
        node.scheduling_config = tpu_v2.SchedulingConfig()
        node.scheduling_config.preemptible = True
    
    node.network_config = tpu_v2.NetworkConfig()
    node.network_config.network = "default"
    node.network_config.subnetwork = "default"
    node.network_config.enable_external_ips = True
    
    print(f"Creating {tpu_type} TPU in {zone} (preemptible={preemptible})...")
    
    try:
        operation = client.create_node(
            parent=parent,
            node_id=name,
            node=node,
        )
        result = operation.result(timeout=600)
        print(f"✅ TPU created: {result.state.name}")
        if result.network_endpoints:
            for ep in result.network_endpoints:
                if ep.access_config:
                    print(f"   External IP: {ep.access_config.external_ip}")
        return result
    except exceptions.AlreadyExists:
        print("TPU already exists")
        return client.get_node(name=f"{parent}/nodes/{name}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def list_tpus():
    """List all TPUs in the project."""
    client = get_tpu_client()
    zones = ["us-central1-a", "us-central1-b", "us-central1-f", "us-central2-b",
             "us-west4-a", "europe-west4-a"]
    
    print("TPU VMs in project:")
    found = False
    for zone in zones:
        parent = f"projects/{PROJECT_ID}/locations/{zone}"
        try:
            nodes = list(client.list_nodes(parent=parent))
            for node in nodes:
                found = True
                name = node.name.split("/")[-1]
                ip = ""
                if node.network_endpoints and node.network_endpoints[0].access_config:
                    ip = node.network_endpoints[0].access_config.external_ip
                print(f"  {name} ({node.accelerator_type}) - {node.state.name} - {ip}")
        except:
            pass
    
    if not found:
        print("  No TPUs found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tpu_benchmark.py list          - List TPUs")
        print("  python tpu_benchmark.py create        - Create test TPU")
        print("  python tpu_benchmark.py delete        - Delete test TPU")
        print("  python tpu_benchmark.py script        - Print benchmark script")
        print("  python tpu_benchmark.py ssh           - Print SSH command")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "list":
        list_tpus()
    elif cmd == "create":
        tpu_type = sys.argv[2] if len(sys.argv) > 2 else "v6e-1"
        zone = sys.argv[3] if len(sys.argv) > 3 else "us-central1-b"
        create_tpu(tpu_type=tpu_type, zone=zone)
    elif cmd == "delete":
        delete_tpu()
    elif cmd == "script":
        print(create_benchmark_script())
    elif cmd == "ssh":
        node, zone = find_tpu()
        if node:
            ip = None
            if node.network_endpoints and node.network_endpoints[0].access_config:
                ip = node.network_endpoints[0].access_config.external_ip
            print(f"\nTPU: jaxbench-test in {zone}")
            print(f"External IP: {ip}")
            print(f"\nSSH command:")
            print(f"  gcloud compute tpus tpu-vm ssh jaxbench-test --zone={zone} --project={PROJECT_ID}")
            print(f"\nOr run benchmark directly:")
            print(f"  gcloud compute tpus tpu-vm ssh jaxbench-test --zone={zone} --project={PROJECT_ID} --command='python3 -c \"$(cat benchmark.py)\"'")
        else:
            print("No TPU found. Run 'python tpu_benchmark.py create' first.")
    else:
        print(f"Unknown command: {cmd}")

