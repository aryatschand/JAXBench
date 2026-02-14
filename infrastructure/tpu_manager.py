"""
TPU Manager for JAXBench.

Handles TPU VM lifecycle and SSH execution on Google Cloud TPUs.
"""

import os
import time
import subprocess
import logging
from typing import Optional

from google.oauth2 import service_account
from google.cloud import tpu_v2, storage
from google.api_core import exceptions

logger = logging.getLogger("jaxbench")

# Configuration - paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, "credentials.json")
PROJECT_ID = "jaxbench"
ZONE = "us-central1-b"
BUCKET_NAME = "tpu-dumps"
SSH_KEY = os.path.expanduser("~/.ssh/id_rsa_tpu")
SSH_USER = "REDACTED_SSH_USER"


def get_credentials():
    """Get GCP credentials from service account file."""
    return service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


class TPUManager:
    """Manages TPU VM lifecycle and SSH execution."""
    
    def __init__(self, tpu_name: str = "jaxbench-runner", tpu_type: str = "v6e-1"):
        self.tpu_name = tpu_name
        self.tpu_type = tpu_type
        self.tpu_ip = None
        self.client = tpu_v2.TpuClient(credentials=get_credentials())
        self.storage = storage.Client(credentials=get_credentials(), project=PROJECT_ID)
        self.bucket = self.storage.bucket(BUCKET_NAME)
        self._setup_done = False
    
    def get_or_create_tpu(self) -> str:
        """Get existing TPU or create new one. Returns IP address."""
        node_path = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{self.tpu_name}"
        
        # Check if TPU exists
        try:
            node = self.client.get_node(name=node_path)
            if node.state.name == "READY":
                self.tpu_ip = node.network_endpoints[0].access_config.external_ip
                logger.info(f"TPU {self.tpu_name} already running at {self.tpu_ip}")
                return self.tpu_ip
            elif node.state.name in ["CREATING", "STARTING", "RESTARTING"]:
                logger.info(f"TPU is {node.state.name}, waiting...")
                return self._wait_for_tpu()
            elif node.state.name == "PREEMPTED":
                logger.info(f"TPU was preempted, deleting and recreating...")
                self.delete_tpu()
                time.sleep(10)
            elif node.state.name in ["STOPPED", "STOPPING"]:
                logger.info(f"TPU is {node.state.name}, starting...")
                self.client.start_node(name=node_path)
                return self._wait_for_tpu()
            else:
                logger.info(f"TPU in state {node.state.name}, waiting...")
                return self._wait_for_tpu()
        except exceptions.NotFound:
            pass
        
        # Create new TPU
        logger.info(f"Creating TPU {self.tpu_type}...")
        runtime = "v2-alpha-tpuv6e" if self.tpu_type.startswith("v6e") else "tpu-ubuntu2204-base"
        
        node = tpu_v2.Node()
        node.accelerator_type = self.tpu_type
        node.runtime_version = runtime
        node.scheduling_config = tpu_v2.SchedulingConfig()
        node.scheduling_config.preemptible = True
        node.network_config = tpu_v2.NetworkConfig()
        node.network_config.network = "default"
        node.network_config.subnetwork = "default"
        node.network_config.enable_external_ips = True
        
        parent = f"projects/{PROJECT_ID}/locations/{ZONE}"
        start = time.time()
        op = self.client.create_node(parent=parent, node_id=self.tpu_name, node=node)
        result = op.result(timeout=600)
        
        self.tpu_ip = result.network_endpoints[0].access_config.external_ip
        logger.info(f"TPU ready in {time.time()-start:.0f}s at {self.tpu_ip}")
        return self.tpu_ip
    
    def _wait_for_tpu(self, timeout: int = 600) -> str:
        """Wait for TPU to be ready."""
        node_path = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{self.tpu_name}"
        start = time.time()
        
        while time.time() - start < timeout:
            node = self.client.get_node(name=node_path)
            if node.state.name == "READY":
                self.tpu_ip = node.network_endpoints[0].access_config.external_ip
                return self.tpu_ip
            time.sleep(10)
        
        raise TimeoutError("TPU creation timed out")
    
    def setup_environment(self):
        """Install required packages on TPU VM."""
        if self._setup_done:
            return
        
        logger.info("Setting up TPU environment...")
        
        setup_commands = [
            ("pip install -q torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch CPU"),
            ("pip install -q 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html", "Installing torch_xla"),
            ("pip install -q numpy google-cloud-storage", "Installing numpy & GCS"),
            ("pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html", "Installing JAX for TPU"),
        ]
        
        for cmd, desc in setup_commands:
            logger.info(f"  {desc}...")
            result = self.run_ssh(cmd, timeout=300)
            if "error" in result.lower() and "warning" not in result.lower():
                logger.warning(f"    Warning: {result[:100]}")
        
        # Verify installation
        logger.info("  Verifying installation...")
        verify = self.run_ssh("python3 -c 'import jax; import torch; import torch_xla; print(f\"JAX={jax.__version__}, torch={torch.__version__}, torch_xla={torch_xla.__version__}\")'")
        logger.info(f"  Versions: {verify.strip()}")
        
        self._setup_done = True
        logger.info("TPU environment ready")
    
    def run_ssh(self, command: str, timeout: int = 120, verbose: bool = False) -> str:
        """Run command on TPU via SSH."""
        if not self.tpu_ip:
            raise RuntimeError("TPU not initialized")
        
        ssh_cmd = [
            "ssh",
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout=15",
            f"{SSH_USER}@{self.tpu_ip}",
            command
        ]
        
        if verbose:
            logger.info(f"    SSH: {command[:80]}...")
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
            output = result.stdout + result.stderr
            lines = [l for l in output.split('\n') if 'Could not open' not in l]
            return '\n'.join(lines)
        except subprocess.TimeoutExpired:
            logger.warning(f"    SSH timeout after {timeout}s")
            return "ERROR: Command timed out"
        except Exception as e:
            logger.error(f"    SSH error: {e}")
            return f"ERROR: {e}"
    
    def run_python(self, script: str, timeout: int = 120) -> str:
        """Run Python script on TPU."""
        escaped = script.replace("'", "'\"'\"'")
        return self.run_ssh(f"PJRT_DEVICE=TPU python3 -c '{escaped}'", timeout=timeout)
    
    def clear_tpu_state(self):
        """Clear TPU state (kill processes, remove lock files)."""
        self.run_ssh("pkill -9 python3 2>/dev/null; sudo rm -f /tmp/libtpu_lockfile; sleep 1", timeout=15)
    
    def delete_tpu(self):
        """Delete TPU VM and wait for completion."""
        node_path = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{self.tpu_name}"
        try:
            op = self.client.delete_node(name=node_path)
            logger.info("TPU deletion initiated, waiting...")
            op.result(timeout=300)
            logger.info("TPU deleted")
        except exceptions.NotFound:
            logger.info("TPU not found (already deleted)")
        except Exception as e:
            logger.warning(f"TPU deletion error: {e}")
    
    def get_internal_ip(self) -> Optional[str]:
        """Get TPU internal IP address."""
        node_path = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{self.tpu_name}"
        try:
            node = self.client.get_node(name=node_path)
            if node.network_endpoints:
                return node.network_endpoints[0].ip_address
        except:
            pass
        return None
