"""TPU execution utilities for remote Pallas benchmarking."""

import subprocess
import json
import tempfile
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TPUConfig:
    """TPU connection configuration."""
    ip: str
    user: str = "REDACTED_SSH_USER"
    key_path: str = "~/.ssh/id_rsa_tpu"
    zone: str = "us-central1-b"
    project: str = "jaxbench"


DEFAULT_TPU = TPUConfig(ip="REDACTED_IP")


def run_on_tpu(
    script_path: str,
    tpu_config: Optional[TPUConfig] = None,
    args: str = "",
    timeout: int = 600,
) -> dict:
    """
    Copy a script to TPU and run it, returning the results.

    Args:
        script_path: Local path to Python script
        tpu_config: TPU connection config (uses default if None)
        args: Command line arguments for the script
        timeout: Timeout in seconds

    Returns:
        Dict with 'stdout', 'stderr', 'returncode'
    """
    config = tpu_config or DEFAULT_TPU
    key_path = os.path.expanduser(config.key_path)
    remote_path = f"~/{os.path.basename(script_path)}"

    # Copy script to TPU
    scp_cmd = [
        "scp", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        script_path,
        f"{config.user}@{config.ip}:{remote_path}"
    ]

    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {
            'stdout': '',
            'stderr': f"SCP failed: {result.stderr}",
            'returncode': result.returncode,
        }

    # Run script on TPU
    ssh_cmd = [
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        f"{config.user}@{config.ip}",
        f"PJRT_DEVICE=TPU python3 {remote_path} {args}"
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            'stdout': '',
            'stderr': f"Timeout after {timeout}s",
            'returncode': -1,
        }


def run_python_on_tpu(
    code: str,
    tpu_config: Optional[TPUConfig] = None,
    timeout: int = 600,
) -> dict:
    """
    Run Python code directly on TPU.

    Args:
        code: Python code to execute
        tpu_config: TPU connection config
        timeout: Timeout in seconds

    Returns:
        Dict with 'stdout', 'stderr', 'returncode'
    """
    # Write code to temp file
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        return run_on_tpu(temp_path, tpu_config, timeout=timeout)
    finally:
        os.unlink(temp_path)


def get_tpu_info(tpu_config: Optional[TPUConfig] = None) -> dict:
    """Get TPU device information."""
    code = """
import jax
import json
devices = jax.devices()
info = {
    'num_devices': len(devices),
    'devices': [str(d) for d in devices],
    'jax_version': jax.__version__,
}
print(json.dumps(info))
"""
    result = run_python_on_tpu(code, tpu_config, timeout=60)
    if result['returncode'] == 0:
        try:
            return json.loads(result['stdout'].strip())
        except json.JSONDecodeError:
            pass
    return {'error': result['stderr'] or result['stdout']}


def check_tpu_connection(tpu_config: Optional[TPUConfig] = None) -> bool:
    """Check if TPU is accessible."""
    config = tpu_config or DEFAULT_TPU
    key_path = os.path.expanduser(config.key_path)

    cmd = [
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{config.user}@{config.ip}",
        "echo ok"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False
