"""
GCP TPU Test Script for JAXBench.

Tests TPU allocation and JAX execution on Google Cloud TPUs.

Usage:
    python gcp_tpu_test.py
"""

import os
import json
import subprocess
import time
from google.cloud import tpu_v2
from google.oauth2 import service_account
from google.api_core import exceptions

# Configuration
PROJECT_ID = "jaxbench"
ZONE = "us-central1-a"  # TPU availability zone
TPU_NAME = "jaxbench-test-tpu"
CREDENTIALS_FILE = "credentials.json"

# TPU configurations (cheapest options)
# v2-8 is the cheapest, followed by v3-8
# Preemptible/Spot instances are much cheaper
TPU_CONFIGS = {
    "v2-8": {"accelerator_type": "v2-8", "runtime_version": "tpu-vm-tf-2.16.1-pjrt"},
    "v3-8": {"accelerator_type": "v3-8", "runtime_version": "tpu-vm-tf-2.16.1-pjrt"},
    "v4-8": {"accelerator_type": "v4-8", "runtime_version": "tpu-vm-tf-2.16.1-pjrt"},
    "v5litepod-1": {"accelerator_type": "v5litepod-1", "runtime_version": "v2-alpha-tpuv5-lite"},
    "v5litepod-4": {"accelerator_type": "v5litepod-4", "runtime_version": "v2-alpha-tpuv5-lite"},
}

# Zones with TPU availability
TPU_ZONES = {
    "v2-8": ["us-central1-b", "us-central1-c", "us-central1-f", "europe-west4-a"],
    "v3-8": ["us-central1-a", "us-central1-b", "us-central1-f", "europe-west4-a"],
    "v4-8": ["us-central2-b"],
    "v5litepod-1": ["us-west4-a", "us-east5-a"],
    "v5litepod-4": ["us-west4-a", "us-east5-a"],
}


def get_credentials():
    """Load GCP credentials from service account file."""
    creds_path = os.path.join(os.path.dirname(__file__), CREDENTIALS_FILE)
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return credentials


def get_tpu_client():
    """Create TPU client with credentials."""
    credentials = get_credentials()
    return tpu_v2.TpuClient(credentials=credentials)


def list_available_tpus():
    """List available TPU types and their locations."""
    print("=" * 70)
    print("Checking TPU availability...")
    print("=" * 70)
    
    client = get_tpu_client()
    
    # List accelerator types
    try:
        parent = f"projects/{PROJECT_ID}/locations/-"
        accelerator_types = client.list_accelerator_types(parent=parent)
        
        print("\nAvailable TPU accelerator types:")
        for acc_type in accelerator_types:
            print(f"  - {acc_type.type_}: {acc_type.name}")
    except Exception as e:
        print(f"Error listing accelerator types: {e}")
    
    # List runtime versions
    try:
        runtime_versions = client.list_runtime_versions(parent=parent)
        print("\nAvailable runtime versions (first 10):")
        for i, version in enumerate(runtime_versions):
            if i >= 10:
                print("  ... (more available)")
                break
            print(f"  - {version.name.split('/')[-1]}")
    except Exception as e:
        print(f"Error listing runtime versions: {e}")


def list_existing_tpus():
    """List existing TPU VMs in the project."""
    print("\n" + "=" * 70)
    print("Existing TPU VMs in project...")
    print("=" * 70)
    
    client = get_tpu_client()
    
    # Check multiple zones
    zones_to_check = ["us-central1-a", "us-central1-b", "us-central1-f", 
                      "us-central2-b", "europe-west4-a", "us-west4-a", "us-east5-a"]
    
    found_any = False
    for zone in zones_to_check:
        try:
            parent = f"projects/{PROJECT_ID}/locations/{zone}"
            nodes = client.list_nodes(parent=parent)
            for node in nodes:
                found_any = True
                print(f"\n  TPU: {node.name.split('/')[-1]}")
                print(f"    Zone: {zone}")
                print(f"    Type: {node.accelerator_type}")
                print(f"    State: {node.state.name}")
                if node.network_endpoints:
                    for ep in node.network_endpoints:
                        print(f"    IP: {ep.ip_address}")
        except exceptions.NotFound:
            pass
        except Exception as e:
            if "404" not in str(e):
                print(f"  Error checking {zone}: {e}")
    
    if not found_any:
        print("  No existing TPU VMs found.")


def create_tpu_vm(tpu_type="v2-8", zone=None, preemptible=True):
    """
    Create a TPU VM.
    
    Args:
        tpu_type: TPU type (v2-8, v3-8, v4-8, etc.)
        zone: GCP zone (auto-selected if None)
        preemptible: Use preemptible/spot instances (cheaper)
    """
    if zone is None:
        zone = TPU_ZONES.get(tpu_type, ["us-central1-b"])[0]
    
    config = TPU_CONFIGS.get(tpu_type, TPU_CONFIGS["v2-8"])
    
    print("\n" + "=" * 70)
    print(f"Creating TPU VM: {TPU_NAME}")
    print("=" * 70)
    print(f"  Type: {tpu_type}")
    print(f"  Zone: {zone}")
    print(f"  Runtime: {config['runtime_version']}")
    print(f"  Preemptible: {preemptible}")
    
    client = get_tpu_client()
    parent = f"projects/{PROJECT_ID}/locations/{zone}"
    
    # TPU VM configuration
    node = tpu_v2.Node()
    node.accelerator_type = config["accelerator_type"]
    node.runtime_version = config["runtime_version"]
    
    # Use preemptible for cost savings
    if preemptible:
        node.scheduling_config = tpu_v2.SchedulingConfig()
        node.scheduling_config.preemptible = True
    
    # Network config
    node.network_config = tpu_v2.NetworkConfig()
    node.network_config.enable_external_ips = True
    
    try:
        print("\n  Starting TPU creation...")
        operation = client.create_node(
            parent=parent,
            node_id=TPU_NAME,
            node=node,
        )
        
        print("  Waiting for TPU to be ready (this may take 2-5 minutes)...")
        result = operation.result(timeout=600)  # 10 minute timeout
        
        print(f"\n  ✅ TPU VM created successfully!")
        print(f"  Name: {result.name}")
        print(f"  State: {result.state.name}")
        
        if result.network_endpoints:
            for ep in result.network_endpoints:
                print(f"  IP Address: {ep.ip_address}")
        
        return result
        
    except exceptions.AlreadyExists:
        print(f"  TPU VM '{TPU_NAME}' already exists. Fetching info...")
        node_name = f"{parent}/nodes/{TPU_NAME}"
        return client.get_node(name=node_name)
        
    except Exception as e:
        print(f"  ❌ Failed to create TPU: {e}")
        return None


def delete_tpu_vm(zone=None):
    """Delete the test TPU VM."""
    if zone is None:
        # Try common zones
        zones = ["us-central1-b", "us-central1-a", "us-central1-f", "us-central2-b"]
    else:
        zones = [zone]
    
    client = get_tpu_client()
    
    for z in zones:
        try:
            node_name = f"projects/{PROJECT_ID}/locations/{z}/nodes/{TPU_NAME}"
            print(f"\nDeleting TPU VM: {TPU_NAME} in {z}...")
            operation = client.delete_node(name=node_name)
            operation.result(timeout=300)
            print(f"  ✅ TPU VM deleted successfully!")
            return True
        except exceptions.NotFound:
            continue
        except Exception as e:
            print(f"  Error: {e}")
    
    print("  TPU VM not found in any zone.")
    return False


def run_jax_on_tpu(tpu_ip):
    """
    Run JAX benchmark code on the TPU VM via SSH.
    
    Args:
        tpu_ip: IP address of the TPU VM
    """
    print("\n" + "=" * 70)
    print(f"Running JAX benchmark on TPU ({tpu_ip})")
    print("=" * 70)
    
    # JAX benchmark code to run on TPU
    jax_code = '''
import jax
import jax.numpy as jnp
import time

print("=" * 60)
print("JAX TPU Benchmark")
print("=" * 60)

# Device info
print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Matrix multiplication benchmark
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

sizes = [1024, 2048, 4096]
results = []

for size in sizes:
    print(f"\\nBenchmarking matmul ({size}x{size})...")
    
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (size, size), dtype=jnp.float32)
    b = jax.random.normal(key, (size, size), dtype=jnp.float32)
    
    # Warmup
    c = matmul(a, b)
    c.block_until_ready()
    
    # Benchmark
    num_iters = 100
    start = time.perf_counter()
    for _ in range(num_iters):
        c = matmul(a, b)
        c.block_until_ready()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    tflops = (2 * size**3 * num_iters / elapsed) / 1e12
    
    print(f"  Average time: {avg_ms:.3f} ms")
    print(f"  Throughput: {tflops:.2f} TFLOPS")
    results.append({"size": size, "avg_ms": avg_ms, "tflops": tflops})

print("\\n" + "=" * 60)
print("Results Summary:")
for r in results:
    print(f"  {r['size']}x{r['size']}: {r['avg_ms']:.3f} ms, {r['tflops']:.2f} TFLOPS")
print("=" * 60)
'''
    
    # Create a temporary script file
    script_content = f'''#!/bin/bash
# Install JAX with TPU support
pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Run the benchmark
python3 << 'PYTHON_EOF'
{jax_code}
PYTHON_EOF
'''
    
    print("\nTo run JAX on the TPU, SSH into the VM and execute:")
    print("-" * 50)
    print(f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone=<zone> --project={PROJECT_ID}")
    print("-" * 50)
    print("\nOr use the following command to run directly:")
    print(f'gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone=<zone> --project={PROJECT_ID} --command="pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c \\"{jax_code.replace(chr(10), ";").replace(chr(34), chr(39))}\\"" ')
    
    return script_content


def test_gcp_connection():
    """Test basic GCP connection and permissions."""
    print("=" * 70)
    print("Testing GCP Connection")
    print("=" * 70)
    
    try:
        credentials = get_credentials()
        print(f"  ✅ Credentials loaded from {CREDENTIALS_FILE}")
        print(f"  Project: {PROJECT_ID}")
        print(f"  Service Account: {credentials.service_account_email}")
        
        # Test TPU API access
        client = get_tpu_client()
        print(f"  ✅ TPU client created successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("JAXBench GCP TPU Test")
    print("=" * 70)
    
    # Test connection
    if not test_gcp_connection():
        return
    
    # List what's available
    list_available_tpus()
    list_existing_tpus()
    
    # Try to create a TPU VM
    print("\n" + "-" * 70)
    print("Attempting to create a cheap TPU VM...")
    print("-" * 70)
    
    # Try different TPU types in order of cost (cheapest first)
    tpu_types_to_try = [
        ("v2-8", "us-central1-b"),
        ("v2-8", "us-central1-f"),
        ("v3-8", "us-central1-a"),
        ("v3-8", "us-central1-b"),
    ]
    
    tpu_node = None
    for tpu_type, zone in tpu_types_to_try:
        print(f"\nTrying {tpu_type} in {zone}...")
        try:
            tpu_node = create_tpu_vm(tpu_type=tpu_type, zone=zone, preemptible=True)
            if tpu_node:
                break
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    if tpu_node and tpu_node.network_endpoints:
        ip = tpu_node.network_endpoints[0].ip_address
        run_jax_on_tpu(ip)
        
        # Ask about cleanup
        print("\n" + "-" * 70)
        print("TPU VM created. Don't forget to delete it to avoid charges!")
        print(f"To delete: python gcp_tpu_test.py --delete")
        print("-" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--delete":
        delete_tpu_vm()
    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        test_gcp_connection()
        list_existing_tpus()
    else:
        main()

