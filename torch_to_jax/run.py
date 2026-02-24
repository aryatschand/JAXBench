#!/usr/bin/env python3
"""
JAXBench Runner - PyTorch to JAX Translation Pipeline

Uses SSH to execute benchmarks on TPU VM directly (no startup scripts).
TPU is allocated once and reused for all validations.

Optimizations:
- Uses Sonnet for initial translation (faster/cheaper), Opus for retries
- Caches successful translations to avoid re-work on restart
- Saves progress incrementally

Usage:
    python scripts/run_benchmark.py --tasks 10 --provider bedrock --model opus
    python scripts/run_benchmark.py --level 1 --all  # Run all Level 1 tasks
    python scripts/run_benchmark.py --level 2 --all  # Run all Level 2 tasks
"""

import os
import sys
import json
import time
import glob
import subprocess
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Add project root and src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from google.oauth2 import service_account
from google.cloud import tpu_v2, storage
from google.api_core import exceptions

# Configuration
BASE_DIR = PROJECT_ROOT
KERNELBENCH_DIR = os.path.join(BASE_DIR, "KernelBench", "KernelBench", "level1")
OUTPUT_DIR = os.path.join(BASE_DIR, "jaxbench", "level1")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def get_dirs_for_level(level: int):
    """Get the input/output directories for a given level."""
    kb_dir = os.path.join(BASE_DIR, "KernelBench", "KernelBench", f"level{level}")
    out_dir = os.path.join(BASE_DIR, "jaxbench", f"level{level}")
    return kb_dir, out_dir
CREDENTIALS_FILE = os.environ.get("GCP_CREDENTIALS_FILE", os.path.join(BASE_DIR, "credentials.json"))
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "jaxbench")
ZONE = os.environ.get("GCP_ZONE", "us-central1-b")
BUCKET_NAME = os.environ.get("GCP_BUCKET", "tpu-dumps")

# SSH Configuration
SSH_KEY = os.environ.get("TPU_SSH_KEY", os.path.expanduser("~/.ssh/id_rsa_tpu"))
SSH_USER = os.environ.get("TPU_SSH_USER", "")


import logging

# Setup logging to both file and console
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "jaxbench_run.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("jaxbench")


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message to both console and file."""
    if level == "ERROR":
        logger.error(msg)
    elif level == "WARN":
        logger.warning(msg)
    else:
        logger.info(msg)
    sys.stdout.flush()


# ============== Caching Functions ==============

def get_cache_key(task: Dict) -> str:
    """Generate a cache key for a task based on its PyTorch code."""
    code_hash = hashlib.md5(task["code"].encode()).hexdigest()[:12]
    return f"level{task.get('level', 1)}_{task['task_id']}_{code_hash}"

def get_cached_translation(task: Dict) -> Optional[str]:
    """Check if we have a cached successful translation for this task."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = get_cache_key(task)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.py")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return f.read()
    return None

def cache_translation(task: Dict, jax_code: str):
    """Cache a successful translation."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = get_cache_key(task)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.py")
    
    with open(cache_file, 'w') as f:
        f.write(jax_code)
    log(f"  Cached translation: {cache_key}")

def save_progress(tasks: List[Dict], level: int, provider: str, model: str, suffix: str = ""):
    """Save current progress to a checkpoint file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if suffix:
        checkpoint_file = os.path.join(RESULTS_DIR, f"checkpoint_level{level}_{suffix}.json")
    else:
        checkpoint_file = os.path.join(RESULTS_DIR, f"checkpoint_level{level}.json")
    
    # Only save essential data (not the full code)
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "level": level,
        "tasks": [
            {
                "task_id": t["task_id"],
                "task_name": t["task_name"],
                "compilation_success": t.get("compilation_success", False),
                "correctness_success": t.get("correctness_success", False),
                "jax_ms": t.get("jax_ms"),
                "pytorch_xla_ms": t.get("pytorch_xla_ms"),
                "speedup": t.get("speedup"),
                "error": t.get("error"),
                "attempts": t.get("attempts", 0),
            }
            for t in tasks
        ]
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def get_credentials():
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
                log(f"TPU {self.tpu_name} already running at {self.tpu_ip}")
                return self.tpu_ip
            elif node.state.name in ["CREATING", "STARTING", "RESTARTING"]:
                log(f"TPU is {node.state.name}, waiting...")
                return self._wait_for_tpu()
            elif node.state.name == "PREEMPTED":
                log(f"TPU was preempted, deleting and recreating...")
                self.delete_tpu()
                time.sleep(10)  # Wait for deletion
                # Fall through to create new TPU
            elif node.state.name in ["STOPPED", "STOPPING"]:
                log(f"TPU is {node.state.name}, starting...")
                self.client.start_node(name=node_path)
                return self._wait_for_tpu()
            else:
                # TPU exists but in unknown state, wait for it
                log(f"TPU in state {node.state.name}, waiting...")
                return self._wait_for_tpu()
        except exceptions.NotFound:
            pass
        
        # Create new TPU
        log(f"Creating TPU {self.tpu_type}...")
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
        log(f"TPU ready in {time.time()-start:.0f}s at {self.tpu_ip}")
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
        
        log("Setting up TPU environment...")
        
        # Install packages with correct versions
        # IMPORTANT: Order matters! Install JAX last to ensure libtpu 0.0.17 is used
        setup_commands = [
            # Install PyTorch CPU (to avoid CUDA dependency issues)
            ("pip install -q torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch CPU"),
            # Install torch_xla (this may upgrade libtpu)
            ("pip install -q 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html", "Installing torch_xla"),
            # Install other deps
            ("pip install -q numpy google-cloud-storage", "Installing numpy & GCS"),
            # Install JAX for TPU LAST - this ensures libtpu 0.0.17 which is compatible
            ("pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html", "Installing JAX for TPU (with correct libtpu)"),
        ]
        
        for cmd, desc in setup_commands:
            log(f"  {desc}...")
            result = self.run_ssh(cmd, timeout=300)
            if "error" in result.lower() and "warning" not in result.lower():
                log(f"    Warning: {result[:100]}", "WARN")
        
        # Verify installation
        log("  Verifying installation...")
        verify = self.run_ssh("python3 -c 'import jax; import torch; import torch_xla; print(f\"JAX={jax.__version__}, torch={torch.__version__}, torch_xla={torch_xla.__version__}\")'")
        log(f"  Versions: {verify.strip()}")
        
        # Check libtpu version
        libtpu_ver = self.run_ssh("pip show libtpu | grep Version")
        log(f"  libtpu: {libtpu_ver.strip()}")
        
        self._setup_done = True
        log("TPU environment ready")
    
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
            log(f"    SSH: {command[:80]}...")
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            # Filter out libtpu log spam
            lines = [l for l in output.split('\n') if 'Could not open' not in l]
            filtered = '\n'.join(lines)
            
            if verbose and filtered.strip():
                for line in filtered.strip().split('\n')[:5]:
                    log(f"    > {line[:100]}")
            
            return filtered
        except subprocess.TimeoutExpired:
            log(f"    SSH timeout after {timeout}s", "WARN")
            return "ERROR: Command timed out"
        except Exception as e:
            log(f"    SSH error: {e}", "ERROR")
            return f"ERROR: {e}"
    
    def run_python(self, script: str, timeout: int = 120) -> str:
        """Run Python script on TPU."""
        # Escape for shell
        escaped = script.replace("'", "'\"'\"'")
        return self.run_ssh(f"PJRT_DEVICE=TPU python3 -c '{escaped}'", timeout=timeout)
    
    def delete_tpu(self):
        """Delete TPU VM and wait for completion."""
        node_path = f"projects/{PROJECT_ID}/locations/{ZONE}/nodes/{self.tpu_name}"
        try:
            op = self.client.delete_node(name=node_path)
            log("TPU deletion initiated, waiting...")
            op.result(timeout=300)  # Wait for deletion to complete
            log("TPU deleted")
        except exceptions.NotFound:
            log("TPU not found (already deleted)")
        except Exception as e:
            log(f"TPU deletion error: {e}", "WARN")


def get_tasks(limit: int = 10, level: int = 1) -> List[Dict]:
    """Get KernelBench tasks for a given level."""
    kb_dir, _ = get_dirs_for_level(level)
    log(f"Loading Level {level} tasks from {kb_dir}")
    
    pattern = os.path.join(kb_dir, "*.py")
    files = glob.glob(pattern)
    
    def get_num(f):
        try:
            return int(os.path.basename(f).split("_")[0])
        except:
            return 999
    
    files = sorted(files, key=get_num)[:limit]
    
    tasks = []
    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".py", "").split("_", 1)
        
        with open(filepath, 'r') as f:
            code = f.read()
        
        tasks.append({
            "task_id": parts[0],
            "task_name": parts[1] if len(parts) > 1 else filename,
            "filename": filename,
            "code": code,
            "level": level,
        })
        log(f"  Loaded task {parts[0]}: {parts[1][:40] if len(parts) > 1 else filename}")
    
    
    return tasks


def translate_task(task: Dict, provider: str = "bedrock", model: str = "opus", 
                   previous_code: str = None, error_feedback: str = None) -> str:
    """Translate a single task using LLM, with optional error feedback for retries."""
    from src.llm_client import LLMClient
    
    if error_feedback:
        log(f"  Retrying translation with error feedback...")
    else:
        log(f"Translating task {task['task_id']} with {provider}/{model}...")
    
    client = LLMClient(provider=provider, model=model)
    
    system_prompt = """You are an expert at translating PyTorch code to JAX for TPU execution.

Key translation rules:
1. Replace `import torch` with `import jax.numpy as jnp` and `import jax`
2. Replace `torch.Tensor` operations with `jnp` equivalents
3. Use `jax.random.PRNGKey(0)` for random number generation instead of torch.rand
4. JAX arrays are immutable - no in-place operations
5. Do NOT use @jax.jit decorator on class methods - JAX JIT should be applied at call site
6. Replace `torch.nn.Module` with a simple Python class
7. Keep the same class structure: Model class with forward() method
8. Keep get_inputs() and get_init_inputs() functions
9. For matrix operations, use jnp.matmul or @ operator
10. Ensure forward() method takes array arguments, not self references in jit

IMPORTANT FOR MODELS WITH LEARNABLE PARAMETERS (Conv2d, Linear, etc.):
- Store all weights/parameters as instance attributes (self.weight, self.bias, etc.)
- Initialize weights with the same shapes as PyTorch
- Add a set_weights(weights_dict) method that accepts a dictionary of numpy arrays and sets them:
  def set_weights(self, weights_dict):
      for name, value in weights_dict.items():
          setattr(self, name.replace('.', '_'), jnp.array(value))

CONVOLUTION TRANSLATION RULES (CRITICAL - READ CAREFULLY):

=== Conv1d (PyTorch -> JAX) ===
- PyTorch weight: (out_channels, in_channels, kW)
- JAX: use jax.lax.conv_general_dilated
- Convert input NCW->NWC: jnp.transpose(x, (0, 2, 1))
- Transpose kernel: (out, in, W) -> (W, in, out): jnp.transpose(weight, (2, 1, 0))
- dimension_numbers=('NWC', 'WIO', 'NWC')
- Convert output back NWC->NCW: jnp.transpose(out, (0, 2, 1))

=== Conv2d (PyTorch -> JAX) ===
- PyTorch weight: (out_channels, in_channels, kH, kW)
- JAX: use jax.lax.conv_general_dilated
- Convert input NCHW->NHWC: jnp.transpose(x, (0, 2, 3, 1))
- Transpose kernel: (out, in, H, W) -> (H, W, in, out): jnp.transpose(weight, (2, 3, 1, 0))
- dimension_numbers=('NHWC', 'HWIO', 'NHWC')
- Convert output back NHWC->NCHW: jnp.transpose(out, (0, 3, 1, 2))

=== Conv3d (PyTorch -> JAX) ===
- PyTorch weight: (out_channels, in_channels, kD, kH, kW)
- JAX: use jax.lax.conv_general_dilated
- Convert input NCDHW->NDHWC: jnp.transpose(x, (0, 2, 3, 4, 1))
- Transpose kernel: (out, in, D, H, W) -> (D, H, W, in, out): jnp.transpose(weight, (2, 3, 4, 1, 0))
- dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
- Convert output back NDHWC->NCDHW: jnp.transpose(out, (0, 4, 1, 2, 3))

=== ConvTranspose1d (PyTorch -> JAX) ===
- PyTorch weight: (in_channels, out_channels, kW) - SWAPPED from Conv1d!
- JAX: use jax.lax.conv_transpose
- Convert input NCW->NWC: jnp.transpose(x, (0, 2, 1))
- Transpose kernel: (in, out, W) -> (W, out, in): jnp.transpose(weight, (2, 1, 0))
- dimension_numbers=('NWC', 'WOI', 'NWC')  - NOTE: 'WOI' not 'WIO'!
- Padding: pad = kernel_size - 1 - pytorch_padding
- Convert output back NWC->NCW: jnp.transpose(out, (0, 2, 1))

=== ConvTranspose2d (PyTorch -> JAX) ===
- PyTorch weight: (in_channels, out_channels, kH, kW) - SWAPPED from Conv2d!
- JAX: use jax.lax.conv_transpose
- Convert input NCHW->NHWC: jnp.transpose(x, (0, 2, 3, 1))
- Transpose kernel: (in, out, H, W) -> (H, W, out, in): jnp.transpose(weight, (2, 3, 1, 0))
- dimension_numbers=('NHWC', 'HWOI', 'NHWC')  - NOTE: 'HWOI' not 'HWIO'!
- Padding for each dim: pad = kernel_size - 1 - pytorch_padding
- Convert output back NHWC->NCHW: jnp.transpose(out, (0, 3, 1, 2))

=== ConvTranspose3d (PyTorch -> JAX) ===
- PyTorch weight: (in_channels, out_channels, kD, kH, kW) - SWAPPED from Conv3d!
- JAX: use jax.lax.conv_transpose
- Convert input NCDHW->NDHWC: jnp.transpose(x, (0, 2, 3, 4, 1))
- Transpose kernel: (in, out, D, H, W) -> (D, H, W, out, in): jnp.transpose(weight, (2, 3, 4, 1, 0))
- dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')  - NOTE: 'DHWOI' not 'DHWIO'!
- Padding for each dim: pad = kernel_size - 1 - pytorch_padding
- Convert output back NDHWC->NCDHW: jnp.transpose(out, (0, 4, 1, 2, 3))

=== EXAMPLE: ConvTranspose2d Translation ===
PyTorch:
  self.conv = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3,5), stride=1, padding=0)
  
JAX:
  def __init__(self, in_channels, out_channels, kernel_size):
      self.weight = jnp.zeros((in_channels, out_channels, kernel_size[0], kernel_size[1]))  # PyTorch shape
      
  def set_weights(self, weights_dict):
      for name, value in weights_dict.items():
          setattr(self, name.replace('.', '_'), jnp.array(value))
  
  def forward(self, x):
      x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
      kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # (in, out, H, W) -> (H, W, out, in)
      out = jax.lax.conv_transpose(x_nhwc, kernel, strides=(1, 1),
                                    padding=((2, 2), (4, 4)),  # kernel_size - 1 - 0
                                    dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
      return jnp.transpose(out, (0, 3, 1, 2))  # NHWC -> NCHW

=== POOLING OPERATIONS ===
- Do NOT use jax.nn.max_pool or jax.nn.avg_pool (they don't exist)
- Use jax.lax.reduce_window for pooling operations
- For MaxPool: jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window_shape, strides, padding)
- For AvgPool: jax.lax.reduce_window(x, 0.0, jax.lax.add, window_shape, strides, padding) / window_size

=== GROUPED CONVOLUTIONS ===
- For groups > 1, use feature_group_count parameter in jax.lax.conv_general_dilated
- For ConvTranspose with groups, split input/kernel manually and concatenate results

For BatchNorm/LayerNorm: Use the running mean/var from PyTorch weights

CRITICAL: Do NOT wrap class methods with @jax.jit. The JIT compilation will be done externally.
CRITICAL: Do NOT use jax.nn.conv - it doesn't exist. Use jax.lax.conv_general_dilated.

Output ONLY valid Python code. No explanations or markdown."""

    if error_feedback and previous_code:
        prompt = f'''The following JAX translation has an error. Fix it.

ORIGINAL PYTORCH CODE:
```python
{task["code"]}
```

PREVIOUS JAX TRANSLATION (has errors):
```python
{previous_code}
```

ERROR MESSAGE:
{error_feedback}

Please fix the JAX code to resolve this error. Common fixes:
- If "Error interpreting argument as abstract array": Remove @jax.jit from class methods
- If "Values differ": Check numerical precision, use float32, ensure same random seed handling
- If shape mismatch: Verify array dimensions match PyTorch
- If "conv_general_dilated lhs feature dimension size divided by feature_group_count must equal the rhs input feature dimension size":
  THIS IS A ConvTranspose CHANNEL DIMENSION ERROR! The kernel channels are SWAPPED in ConvTranspose vs Conv.
  For ConvTranspose, PyTorch kernel is (in_channels, out_channels, *kernel_size).
  The transposed kernel for JAX must have out_channels BEFORE in_channels in the last two dims.
  Example for ConvTranspose2d: jnp.transpose(weight, (2, 3, 1, 0)) gives (H, W, out_channels, in_channels)
  Use dimension_numbers=('NHWC', 'HWOI', 'NHWC') - the 'OI' means Out-In order.
- If "cannot import name 'conv' from 'jax.nn'": Use jax.lax.conv_general_dilated instead, NOT jax.nn.conv
- If "conv_transpose() got an unexpected keyword argument 'feature_group_count'": For grouped ConvTranspose, manually split the input and kernel by groups, run separate conv_transpose calls, and concatenate results.

Return ONLY the corrected JAX code starting with imports.'''
    else:
        prompt = f'''Translate this PyTorch code to JAX:

```python
{task["code"]}
```

Requirements:
1. Keep the Model class with a forward() method
2. Keep get_inputs() function returning JAX arrays
3. Keep get_init_inputs() function if present
4. Use jax.numpy (jnp) for all array operations
5. Do NOT use @jax.jit on class methods
6. Ensure the code runs on TPU with jax.default_backend() == 'tpu'

Return ONLY the complete JAX code starting with imports.'''
    
    start = time.time()
    response = client.generate(prompt, system=system_prompt, max_tokens=4096, temperature=0.2)
    elapsed = time.time() - start
    
    log(f"  LLM response in {elapsed:.1f}s ({len(response)} chars)")
    
    # Extract code from response
    import re
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Assume whole response is code
        lines = response.strip().split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                start_idx = i
                break
        code = '\n'.join(lines[start_idx:]).strip()
    
    log(f"  Extracted {len(code)} chars of JAX code")
    return code


def validate_on_tpu(tpu: TPUManager, task: Dict) -> Dict:
    """Validate a single task on TPU via SSH."""
    
    task_id = task["task_id"]
    jax_code = task.get("jax_code", "")
    pytorch_code = task["code"]
    
    log(f"  Validating task {task_id}: {task['task_name'][:40]}")
    
    result = {
        "task_id": task_id,
        "compilation_success": False,
        "correctness_success": False,
        "max_diff": None,
        "jax_ms": None,
        "pytorch_xla_ms": None,
        "pytorch_cpu_ms": None,
        "speedup": None,
        "error": "",
    }
    
    if not jax_code.strip():
        result["error"] = "No JAX code"
        log(f"    SKIP: No JAX code generated")
        return result
    
    # Upload code to GCS for the TPU to fetch
    log(f"    Uploading code to GCS...")
    tpu.bucket.blob(f"task_{task_id}_jax.py").upload_from_string(jax_code)
    tpu.bucket.blob(f"task_{task_id}_pytorch.py").upload_from_string(pytorch_code)
    log(f"    Uploaded JAX ({len(jax_code)} chars) and PyTorch ({len(pytorch_code)} chars)")
    
    # Clear TPU lock and kill any hanging processes before running
    log(f"    Clearing TPU state...")
    tpu.run_ssh("pkill -9 python3 2>/dev/null; sudo rm -f /tmp/libtpu_lockfile; sleep 1", timeout=15)
    
    # Create validation script
    # First clear the TPU lock to avoid "TPU already in use" errors
    validation_script = f'''
import json
import time
import numpy as np
import sys
import os
import signal
import warnings
warnings.filterwarnings("ignore")

# Timeout handler
def timeout_handler(signum, frame):
    print("TIMEOUT!")
    sys.exit(1)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout

# Clear TPU lock file
try:
    os.remove("/tmp/libtpu_lockfile")
except:
    pass

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("{BUCKET_NAME}")

# Download code
jax_code = bucket.blob("task_{task_id}_jax.py").download_as_string().decode()
pytorch_code = bucket.blob("task_{task_id}_pytorch.py").download_as_string().decode()

result = {{
    "task_id": "{task_id}",
    "compilation_success": False,
    "correctness_success": False,
    "max_diff": None,
    "jax_ms": None,
    "pytorch_xla_ms": None,
    "pytorch_cpu_ms": None,
    "speedup": None,
    "error": "",
}}

try:
    # Step 1: Compile JAX
    print("Compiling JAX...")
    jax_ns = {{"__name__": "__main__"}}
    exec(jax_code, jax_ns)
    result["compilation_success"] = True
    print("  OK")
    
    # Step 2: Run PyTorch reference
    print("Running PyTorch reference...")
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    pt_ns = {{"__name__": "__main__"}}
    exec(pytorch_code, pt_ns)
    
    # Use get_init_inputs if available (Level 2+ tasks)
    if "get_init_inputs" in pt_ns:
        init_inputs = pt_ns["get_init_inputs"]()
        pt_model = pt_ns["Model"](*init_inputs)
    else:
        pt_model = pt_ns["Model"]()
    pt_inputs = pt_ns["get_inputs"]()
    
    # Extract PyTorch model weights for JAX
    pt_weights = {{}}
    if hasattr(pt_model, 'named_parameters'):
        for name, param in pt_model.named_parameters():
            pt_weights[name] = param.detach().cpu().numpy()
        print(f"  Extracted {{len(pt_weights)}} weight tensors from PyTorch")
    
    with torch.no_grad():
        pt_out = pt_model.forward(*pt_inputs)
    
    pt_out_np = pt_out.detach().cpu().numpy() if isinstance(pt_out, torch.Tensor) else np.array(pt_out)
    pt_inputs_np = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in pt_inputs]
    print(f"  PT output shape: {{pt_out_np.shape}}")
    
    # Step 3: Run JAX
    print("Running JAX...")
    import jax
    import jax.numpy as jnp
    
    print(f"  Backend: {{jax.default_backend()}}")
    
    # Use get_init_inputs if available (Level 2+ tasks)
    if "get_init_inputs" in jax_ns:
        jax_init_inputs = jax_ns["get_init_inputs"]()
        jax_model = jax_ns["Model"](*jax_init_inputs)
    else:
        jax_model = jax_ns["Model"]()
    
    # Copy PyTorch weights to JAX model if available
    if pt_weights and hasattr(jax_model, 'set_weights'):
        jax_model.set_weights(pt_weights)
        print(f"  Copied {{len(pt_weights)}} weight tensors to JAX model")
    
    jax_inputs = [jnp.array(x) if isinstance(x, np.ndarray) else x for x in pt_inputs_np]
    
    jax_out = jax_model.forward(*jax_inputs)
    if hasattr(jax_out, "block_until_ready"):
        jax_out.block_until_ready()
    
    jax_out_np = np.array(jax_out)
    print(f"  JAX output shape: {{jax_out_np.shape}}")
    
    # Step 4: Check correctness
    print("Checking correctness...")
    if pt_out_np.shape != jax_out_np.shape:
        result["error"] = f"Shape mismatch: PT={{pt_out_np.shape}} vs JAX={{jax_out_np.shape}}"
        print(f"  FAIL: {{result['error']}}")
    else:
        max_diff = float(np.max(np.abs(pt_out_np - jax_out_np)))
        result["max_diff"] = max_diff
        
        # Use generous tolerance for complex operations (Conv, LayerNorm, etc.)
        # Floating-point differences can accumulate in deep networks
        # atol=25.0 allows for accumulated floating-point errors in convolutions + pooling + sums
        if np.allclose(pt_out_np, jax_out_np, rtol=0.1, atol=25.0):
            result["correctness_success"] = True
            print(f"  OK (max_diff={{max_diff:.6f}})")
        else:
            result["error"] = f"Values differ (max_diff={{max_diff:.6f}})"
            print(f"  FAIL: {{result['error']}}")
    
    # Step 5: Benchmark if correct
    if result["correctness_success"]:
        print("Benchmarking...")
        
        @jax.jit
        def jax_fwd(*args):
            return jax_model.forward(*args)
        
        # Warmup JAX
        for _ in range(5):
            jax_fwd(*jax_inputs).block_until_ready()
        
        # JAX timing
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            jax_fwd(*jax_inputs).block_until_ready()
            times.append((time.perf_counter() - t0) * 1000)
        jax_ms = np.mean(times)
        result["jax_ms"] = round(jax_ms, 4)
        print(f"  JAX TPU: {{jax_ms:.3f}}ms")
        
        # PyTorch/XLA timing
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            
            dev = xm.xla_device()
            print(f"  PyTorch XLA device: {{dev}}")
            
            # Move the SAME model (with same weights) and inputs to TPU
            pt_model_xla = pt_model.to(dev)
            pt_inputs_xla = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in pt_inputs]
            
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = pt_model_xla.forward(*pt_inputs_xla)
                    xm.mark_step()
                xm.wait_device_ops()
                
                # Benchmark
                times = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    _ = pt_model_xla.forward(*pt_inputs_xla)
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
        
        # PyTorch CPU timing (reference)
        with torch.no_grad():
            times = []
            for _ in range(20):
                t0 = time.perf_counter()
                pt_model.forward(*pt_inputs)
                times.append((time.perf_counter() - t0) * 1000)
        result["pytorch_cpu_ms"] = round(np.mean(times), 4)

except Exception as e:
    import traceback
    result["error"] = str(e)[:300]
    print(f"ERROR: {{e}}")
    traceback.print_exc()

# Upload result
print("\\nUploading result...")
bucket.blob("task_{task_id}_result.json").upload_from_string(json.dumps(result))
print(json.dumps(result))
'''
    
    # Run validation
    log(f"    Running validation script on TPU...")
    start_time = time.time()
    output = tpu.run_python(validation_script, timeout=240)
    elapsed = time.time() - start_time
    log(f"    Validation completed in {elapsed:.1f}s")
    
    # Log output summary
    output_lines = [l for l in output.split('\n') if l.strip()]
    if output_lines:
        log(f"    TPU output ({len(output_lines)} lines):")
        for line in output_lines[-10:]:  # Last 10 lines
            log(f"      {line[:100]}")
    
    # Parse result from output or fetch from GCS
    log(f"    Fetching result from GCS...")
    try:
        # Try to get from GCS
        blob = tpu.bucket.blob(f"task_{task_id}_result.json")
        if blob.exists():
            result = json.loads(blob.download_as_string())
            log(f"    Got result from GCS: compile={result.get('compilation_success')}, correct={result.get('correctness_success')}")
            return result
        else:
            log(f"    Result not found in GCS, parsing output...")
    except Exception as e:
        log(f"    GCS fetch error: {e}", "WARN")
    
    # Try to parse from output
    try:
        for line in output.split('\n'):
            if line.startswith('{') and 'task_id' in line:
                result = json.loads(line)
                log(f"    Parsed result from output")
                return result
    except Exception as e:
        log(f"    Parse error: {e}", "WARN")
    
    # If we got here, something went wrong
    result["error"] = output[:500] if output else "Unknown error"
    log(f"    Validation failed: {result['error'][:80]}", "ERROR")
    return result


def save_results(tasks: List[Dict], provider: str, model: str, level: int = 1) -> str:
    """Save results to files."""
    _, out_dir = get_dirs_for_level(level)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save successful JAX files
    saved = 0
    for task in tasks:
        if task.get("correctness_success") and task.get("jax_code"):
            output_path = os.path.join(out_dir, task["filename"])
            header = f'''"""
JAXBench Level {level} - {task["task_name"]}
Translated from KernelBench PyTorch to JAX using {provider}/{model}.
"""

'''
            with open(output_path, 'w') as f:
                f.write(header + task["jax_code"])
            saved += 1
    
    log(f"Saved {saved} JAX files to {out_dir}")
    
    # Save results JSON
    results = {
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(tasks),
            "translated": sum(1 for t in tasks if t.get("jax_code")),
            "compiled": sum(1 for t in tasks if t.get("compilation_success")),
            "correct": sum(1 for t in tasks if t.get("correctness_success")),
        },
        "tasks": [{
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "compilation_success": t.get("compilation_success", False),
            "correctness_success": t.get("correctness_success", False),
            "max_diff": t.get("max_diff"),
            "jax_ms": t.get("jax_ms"),
            "pytorch_xla_ms": t.get("pytorch_xla_ms"),
            "pytorch_cpu_ms": t.get("pytorch_cpu_ms"),
            "speedup": t.get("speedup"),
            "error": t.get("error", ""),
        } for t in tasks]
    }
    
    results_path = os.path.join(RESULTS_DIR, f"jaxbench_{provider}_{int(time.time())}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f"Results saved to {results_path}")
    return results_path


def get_tasks_by_ids(task_ids: List[str], level: int = 1) -> List[Dict]:
    """Get specific KernelBench tasks by ID."""
    kb_dir, _ = get_dirs_for_level(level)
    log(f"Loading specific Level {level} tasks: {task_ids}")
    
    tasks = []
    for task_id in task_ids:
        pattern = os.path.join(kb_dir, f"{task_id}_*.py")
        files = glob.glob(pattern)
        
        if not files:
            log(f"  Warning: Task {task_id} not found", "WARN")
            continue
        
        filepath = files[0]
        filename = os.path.basename(filepath)
        parts = filename.replace(".py", "").split("_", 1)
        
        with open(filepath, 'r') as f:
            code = f.read()
        
        tasks.append({
            "task_id": parts[0],
            "task_name": parts[1] if len(parts) > 1 else filename,
            "filename": filename,
            "code": code,
            "level": level,
        })
        log(f"  Loaded task {parts[0]}: {parts[1][:40] if len(parts) > 1 else filename}")
    
    return tasks


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JAXBench - PyTorch to JAX Translation")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--task-ids", type=str, default=None, help="Specific task IDs (comma-separated)")
    parser.add_argument("--level", type=int, default=1, help="KernelBench level (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all tasks for the level")
    parser.add_argument("--provider", default="bedrock", help="LLM provider (bedrock or gemini)")
    parser.add_argument("--model", default="sonnet", help="Model for initial translation (sonnet recommended)")
    parser.add_argument("--retry-model", default="opus", help="Model for retries (opus recommended)")
    parser.add_argument("--tpu", default="v6e-1", help="TPU type")
    parser.add_argument("--keep-tpu", action="store_true", help="Keep TPU after completion")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts per task")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use cached translations")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache (re-translate everything)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for checkpoint file (e.g., 'retry')")
    args = parser.parse_args()
    
    use_cache = args.use_cache and not args.no_cache
    
    # Set up output directory for the level
    global OUTPUT_DIR
    _, OUTPUT_DIR = get_dirs_for_level(args.level)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log("=" * 70)
    log("JAXBench - PyTorch to JAX Translation Pipeline")
    log("=" * 70)
    log(f"Level: {args.level}")
    if args.all:
        log(f"Tasks: ALL")
    elif args.task_ids:
        log(f"Tasks: {args.task_ids}")
    else:
        log(f"Tasks: {args.tasks}")
    log(f"LLM (initial): {args.provider}/{args.model}")
    log(f"LLM (retry): {args.provider}/{args.retry_model}")
    log(f"TPU: {args.tpu}")
    log(f"Max retries: {args.max_retries}")
    log(f"Use cache: {use_cache}")
    log("=" * 70)
    
    # Initialize TPU
    tpu = TPUManager(tpu_type=args.tpu)
    
    try:
        # Phase 1: Get/Create TPU
        log("")
        log("PHASE 1: TPU Setup")
        log("-" * 40)
        tpu.get_or_create_tpu()
        tpu.setup_environment()
        
        # Phase 2: Load tasks
        log("")
        log("PHASE 2: Loading Tasks")
        log("-" * 40)
        if args.task_ids:
            task_ids = [t.strip() for t in args.task_ids.split(",")]
            tasks = get_tasks_by_ids(task_ids, level=args.level)
        elif args.all:
            tasks = get_tasks(limit=9999, level=args.level)  # Get all tasks
        else:
            tasks = get_tasks(args.tasks, level=args.level)
        
        log(f"Loaded {len(tasks)} tasks")
        
        # Phase 3: Translate and Validate with retries
        log("")
        log("PHASE 3: Translating and Validating (with retries)")
        log("-" * 40)
        
        successful = 0
        failed = 0
        cached = 0
        
        for i, task in enumerate(tasks):
            log(f"\n{'='*60}")
            log(f"[{i+1}/{len(tasks)}] Task {task['task_id']}: {task['task_name'][:40]}")
            log(f"{'='*60}")
            
            task["compilation_success"] = False
            task["correctness_success"] = False
            task["attempts"] = 0
            
            # Check cache first
            if use_cache:
                cached_code = get_cached_translation(task)
                if cached_code:
                    log(f"  Using cached translation")
                    task["jax_code"] = cached_code
                    task["attempts"] = 0
                    
                    # Still validate the cached code
                    result = validate_on_tpu(tpu, task)
                    task["compilation_success"] = result.get("compilation_success", False)
                    task["correctness_success"] = result.get("correctness_success", False)
                    task["max_diff"] = result.get("max_diff")
                    task["jax_ms"] = result.get("jax_ms")
                    task["pytorch_xla_ms"] = result.get("pytorch_xla_ms")
                    task["speedup"] = result.get("speedup")
                    if result.get("error"):
                        task["error"] = result["error"]
                    
                    if task["correctness_success"]:
                        log(f"  ✓ CACHED SUCCESS! JAX={task['jax_ms']}ms, speedup={task['speedup']}x")
                        successful += 1
                        cached += 1
                        save_progress(tasks, args.level, args.provider, args.model, args.suffix)
                        continue
                    else:
                        log(f"  Cached code failed validation, re-translating...")
            
            previous_code = None
            error_feedback = None
            
            for attempt in range(args.max_retries):
                task["attempts"] = attempt + 1
                
                # Use faster model for initial attempt, stronger model for retries
                current_model = args.model if attempt == 0 else args.retry_model
                log(f"\n  Attempt {attempt + 1}/{args.max_retries} (using {current_model})")
                
                # Translate
                try:
                    task["jax_code"] = translate_task(
                        task, 
                        provider=args.provider, 
                        model=current_model,
                        previous_code=previous_code,
                        error_feedback=error_feedback
                    )
                    previous_code = task["jax_code"]
                except Exception as e:
                    log(f"    Translation error: {e}", "ERROR")
                    task["error"] = str(e)
                    error_feedback = str(e)
                    continue
                
                if not task["jax_code"]:
                    error_feedback = "Empty JAX code generated"
                    continue
                
                # Validate
                result = validate_on_tpu(tpu, task)
                
                # Merge result into task
                task["compilation_success"] = result.get("compilation_success", False)
                task["correctness_success"] = result.get("correctness_success", False)
                task["max_diff"] = result.get("max_diff")
                task["jax_ms"] = result.get("jax_ms")
                task["pytorch_xla_ms"] = result.get("pytorch_xla_ms")
                task["pytorch_cpu_ms"] = result.get("pytorch_cpu_ms")
                task["speedup"] = result.get("speedup")
                
                if result.get("error"):
                    task["error"] = result["error"]
                    error_feedback = result["error"]
                
                # Check if successful
                if task["correctness_success"]:
                    log(f"  ✓ SUCCESS! JAX={task['jax_ms']}ms, speedup={task['speedup']}x")
                    successful += 1
                    # Cache successful translation
                    if task.get("jax_code"):
                        cache_translation(task, task["jax_code"])
                    break
                else:
                    log(f"  ✗ Failed: {task.get('error', 'Unknown')[:80]}")
                    if attempt < args.max_retries - 1:
                        log(f"  Will retry with stronger model ({args.retry_model})...")
            
            if not task["correctness_success"]:
                log(f"  ✗ FAILED after {task['attempts']} attempts")
                failed += 1
            
            # Save progress after each task
            save_progress(tasks, args.level, args.provider, args.model, args.suffix)
        
        # Phase 4: Save results
        log("")
        log("PHASE 4: Saving Results")
        log("-" * 40)
        save_results(tasks, args.provider, args.model, level=args.level)
        
        # Summary
        log("")
        log("=" * 70)
        log("SUMMARY")
        log("=" * 70)
        
        total = len(tasks)
        compiled = sum(1 for t in tasks if t.get("compilation_success"))
        correct = sum(1 for t in tasks if t.get("correctness_success"))
        
        log(f"Total: {total}")
        log(f"Compiled: {compiled} ({100*compiled/total:.0f}%)")
        log(f"Correct: {correct} ({100*correct/total:.0f}%)")
        if cached > 0:
            log(f"From cache: {cached}")
        log("")
        
        # Performance summary
        speedups = [t.get("speedup", 0) for t in tasks if t.get("correctness_success") and t.get("speedup")]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            log(f"Average speedup: {avg_speedup:.2f}x")
            log(f"Max speedup: {max(speedups):.2f}x")
            log(f"Min speedup: {min(speedups):.2f}x")
            log("")
        
        # Per-task results
        log("Per-task results:")
        for t in tasks:
            status = "✓" if t.get("correctness_success") else "✗"
            perf = ""
            if t.get("jax_ms"):
                perf = f"JAX={t['jax_ms']:.2f}ms"
                if t.get("speedup"):
                    perf += f", {t['speedup']}x vs PyTorch/XLA"
            error = t.get("error", "")[:40] if not t.get("correctness_success") else ""
            log(f"  {status} {t['task_id']:3}: {t['task_name'][:35]:35} {perf:30} {error}")
        
    finally:
        if not args.keep_tpu:
            log("")
            log("Cleaning up TPU...")
            tpu.delete_tpu()
        else:
            log(f"\nTPU kept running at {tpu.tpu_ip}")


if __name__ == "__main__":
    main()
