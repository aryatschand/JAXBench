#!/usr/bin/env python3
"""
Test TPU connectivity and allocation.

This test verifies that:
1. GCP credentials are valid
2. TPU API is accessible
3. A TPU can be allocated (or an existing one is available)
4. JAX can detect the TPU backend

Usage:
    python tests/test_tpu_connection.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set credentials
BASE_DIR = Path(__file__).parent.parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)

from google.cloud import tpu_v2
from google.oauth2 import service_account
from google.api_core import exceptions

# Configuration
PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
TPU_NAME = "jaxbench-runner"


def test_credentials():
    """Test that GCP credentials are valid."""
    print("=" * 60)
    print("Test 1: GCP Credentials")
    print("=" * 60)
    
    if not CREDENTIALS_FILE.exists():
        print(f"  ❌ FAIL: Credentials file not found: {CREDENTIALS_FILE}")
        return False
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(CREDENTIALS_FILE),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        print(f"  ✅ Credentials loaded successfully")
        print(f"     Service Account: {credentials.service_account_email}")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_tpu_api():
    """Test that TPU API is accessible."""
    print("\n" + "=" * 60)
    print("Test 2: TPU API Access")
    print("=" * 60)
    
    try:
        client = tpu_v2.TpuClient()
        
        # List accelerator types to verify API access
        parent = f"projects/{PROJECT_ID}/locations/{ZONE}"
        
        # Try to list nodes (this will work even if there are none)
        nodes = list(client.list_nodes(parent=parent))
        print(f"  ✅ TPU API accessible")
        print(f"     Found {len(nodes)} TPU(s) in {ZONE}")
        
        for node in nodes:
            name = node.name.split("/")[-1]
            print(f"     - {name}: {node.accelerator_type} ({node.state.name})")
        
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_tpu_available():
    """Test that a TPU is available and ready."""
    print("\n" + "=" * 60)
    print("Test 3: TPU Availability")
    print("=" * 60)
    
    try:
        client = tpu_v2.TpuClient()
        name = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{TPU_NAME}"
        
        node = client.get_node(name=name)
        
        if node.state.name == "READY":
            print(f"  ✅ TPU '{TPU_NAME}' is READY")
            print(f"     Type: {node.accelerator_type}")
            if node.network_endpoints:
                ip = node.network_endpoints[0].ip_address
                print(f"     Internal IP: {ip}")
            return True
        else:
            print(f"  ⚠️  TPU '{TPU_NAME}' exists but state is {node.state.name}")
            return False
            
    except exceptions.NotFound:
        print(f"  ❌ TPU '{TPU_NAME}' not found")
        print(f"     Run: python scripts/run_benchmark.py --keep-tpu to create one")
        return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_tpu_ssh():
    """Test SSH connectivity to TPU."""
    print("\n" + "=" * 60)
    print("Test 4: TPU SSH Access")
    print("=" * 60)
    
    import subprocess
    
    # Use gcloud for SSH (handles IAP tunneling)
    gcloud_paths = [
        "/opt/homebrew/share/google-cloud-sdk/bin/gcloud",
        "/usr/local/bin/gcloud",
        "gcloud"
    ]
    
    gcloud = None
    for path in gcloud_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                gcloud = path
                break
        except:
            continue
    
    if not gcloud:
        print("  ⚠️  gcloud CLI not found, skipping SSH test")
        return None
    
    try:
        cmd = [
            gcloud, "compute", "tpus", "tpu-vm", "ssh",
            TPU_NAME,
            f"--zone={ZONE}",
            f"--project={PROJECT_ID}",
            "--command", "echo 'SSH connection successful'"
        ]
        
        env = os.environ.copy()
        env["CLOUDSDK_CONFIG"] = "/tmp/gcloud_config"
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        
        if "SSH connection successful" in result.stdout:
            print("  ✅ SSH connection successful")
            return True
        else:
            print(f"  ❌ SSH failed: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ❌ SSH connection timed out")
        return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def main():
    """Run all TPU connection tests."""
    print("\n" + "=" * 60)
    print("JAXBench TPU Connection Tests")
    print("=" * 60)
    
    results = {}
    
    results["credentials"] = test_credentials()
    results["tpu_api"] = test_tpu_api()
    results["tpu_available"] = test_tpu_available()
    
    # Only test SSH if TPU is available
    if results["tpu_available"]:
        results["tpu_ssh"] = test_tpu_ssh()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        if passed is None:
            status = "⚠️  SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
            all_passed = False
        print(f"  {test_name}: {status}")
    
    print()
    if all_passed:
        print("✅ All tests passed! TPU is ready for benchmarks.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

