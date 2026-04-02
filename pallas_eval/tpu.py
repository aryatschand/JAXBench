"""Lightweight SSH/SCP helper for TPU access."""

import os
import subprocess
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("pallas_eval.tpu")

SSH_KEY = os.path.expanduser(os.environ.get("SSH_KEY", "~/.ssh/id_rsa_tpu"))
SSH_USER = os.environ.get("SSH_USER", "aryatschand")
TPU_IP = os.environ.get("TPU_IP", "")

SSH_OPTS = [
    "-i", SSH_KEY,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=15",
]


def run_ssh(command: str, ip: str | None = None, timeout: int = 120) -> str:
    ip = ip or TPU_IP
    if not ip:
        raise RuntimeError("TPU_IP not set. Export it or pass ip= argument.")
    cmd = ["ssh", *SSH_OPTS, f"{SSH_USER}@{ip}", command]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
        return "\n".join(
            l for l in output.split("\n") if "Could not open" not in l
        )
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out"
    except Exception as e:
        return f"ERROR: {e}"


def scp_to_tpu(local_path: str, remote_path: str, ip: str | None = None, timeout: int = 30):
    ip = ip or TPU_IP
    if not ip:
        raise RuntimeError("TPU_IP not set.")
    cmd = ["scp", *SSH_OPTS, local_path, f"{SSH_USER}@{ip}:{remote_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def scp_from_tpu(remote_path: str, local_path: str, ip: str | None = None, timeout: int = 30):
    ip = ip or TPU_IP
    if not ip:
        raise RuntimeError("TPU_IP not set.")
    cmd = ["scp", *SSH_OPTS, f"{SSH_USER}@{ip}:{remote_path}", local_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def clear_tpu_state(ip: str | None = None):
    run_ssh("pkill -9 python3 2>/dev/null; sudo rm -f /tmp/libtpu_lockfile; sleep 1",
            ip=ip, timeout=15)
