"""
TPU SSH Configuration for JAXBench

This module provides SSH access to GCP TPU VMs for running benchmarks.
The TPU is kept running to avoid spin-up delays.

Usage:
    from tpu_ssh_config import run_on_tpu
    
    result = run_on_tpu("python3 -c 'import jax; print(jax.devices())'")
"""

import subprocess
import os

# TPU Configuration
TPU_IP = "34.171.48.121"  # Current TPU VM IP
TPU_USER = "REDACTED_SSH_USER"
SSH_KEY = os.path.expanduser("~/.ssh/id_rsa_tpu")

# Package versions that work together on TPU v6e
WORKING_VERSIONS = {
    "libtpu": "0.0.17",
    "jax": "0.6.2", 
    "torch": "2.9.0+cpu",
    "torch_xla": "2.9.0",
}

def run_on_tpu(command: str, timeout: int = 300) -> str:
    """Run a command on the TPU VM via SSH."""
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout=15",
        f"{TPU_USER}@{TPU_IP}",
        command
    ]
    
    result = subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    
    return result.stdout + result.stderr


def run_benchmark_script(script: str, timeout: int = 300) -> str:
    """Run a Python script on the TPU VM."""
    # Escape single quotes in script
    script = script.replace("'", "'\"'\"'")
    
    command = f"PJRT_DEVICE=TPU python3 -c '{script}'"
    return run_on_tpu(command, timeout)


if __name__ == "__main__":
    # Test connection
    print("Testing TPU connection...")
    result = run_on_tpu("python3 -c 'import jax; print(f\"JAX backend: {jax.default_backend()}\")'")
    print(result)

