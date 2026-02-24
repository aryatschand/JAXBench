"""
JAX Code Validator on GCP TPU.

Validates:
1. Compilation - Does the JAX code compile?
2. Correctness - Do outputs match PyTorch reference?
3. Performance - Is JAX faster than PyTorch on TPU?
"""

import json
import time
import os
from typing import Dict, Any, Optional, Tuple
from google.oauth2 import service_account
from google.cloud import tpu_v2, storage
from google.api_core import exceptions

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "jaxbench")
ZONE = os.environ.get("GCP_ZONE", "us-central1-b")
BUCKET_NAME = os.environ.get("GCP_BUCKET", "tpu-dumps")
CREDENTIALS_FILE = os.environ.get("GCP_CREDENTIALS_FILE",
                                   os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json"))


def get_credentials():
    """Load GCP credentials."""
    return service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


def get_tpu_client():
    """Get TPU client."""
    return tpu_v2.TpuClient(credentials=get_credentials())


def get_storage_client():
    """Get storage client."""
    return storage.Client(credentials=get_credentials(), project=PROJECT_ID)


class TPUValidator:
    """Validates JAX code on GCP TPU."""
    
    def __init__(self, tpu_type: str = "v6e-1", zone: str = ZONE):
        self.tpu_type = tpu_type
        self.zone = zone
        self.tpu_name = "jaxbench-validator"
        self.client = get_tpu_client()
        self.storage = get_storage_client()
        self.bucket = self.storage.bucket(BUCKET_NAME)
    
    def _get_runtime_version(self) -> str:
        """Get runtime version for TPU type."""
        if self.tpu_type.startswith("v6e"):
            return "v2-alpha-tpuv6e"
        elif self.tpu_type.startswith("v5litepod"):
            return "v2-alpha-tpuv5-lite"
        else:
            return "tpu-ubuntu2204-base"
    
    def create_tpu(self) -> bool:
        """Create TPU VM if not exists."""
        parent = f"projects/{PROJECT_ID}/locations/{self.zone}"
        node_path = f"{parent}/nodes/{self.tpu_name}"
        
        # Check if exists
        try:
            node = self.client.get_node(name=node_path)
            if node.state.name == "READY":
                print(f"  TPU {self.tpu_name} already ready")
                return True
            elif node.state.name == "CREATING":
                print(f"  TPU {self.tpu_name} is creating, waiting...")
                # Wait for it
                for _ in range(60):
                    node = self.client.get_node(name=node_path)
                    if node.state.name == "READY":
                        return True
                    time.sleep(5)
                return False
        except exceptions.NotFound:
            pass
        
        # Create new TPU
        print(f"  Creating TPU {self.tpu_type} in {self.zone}...")
        
        node = tpu_v2.Node()
        node.accelerator_type = self.tpu_type
        node.runtime_version = self._get_runtime_version()
        node.scheduling_config = tpu_v2.SchedulingConfig()
        node.scheduling_config.preemptible = True
        node.network_config = tpu_v2.NetworkConfig()
        node.network_config.network = "default"
        node.network_config.subnetwork = "default"
        node.network_config.enable_external_ips = True
        
        try:
            operation = self.client.create_node(
                parent=parent,
                node_id=self.tpu_name,
                node=node,
            )
            result = operation.result(timeout=600)
            return result.state.name == "READY"
        except exceptions.AlreadyExists:
            return True
        except Exception as e:
            print(f"  Failed to create TPU: {e}")
            return False
    
    def delete_tpu(self):
        """Delete TPU VM."""
        node_path = f"projects/{PROJECT_ID}/locations/{self.zone}/nodes/{self.tpu_name}"
        try:
            operation = self.client.delete_node(name=node_path)
            # Don't wait, let it delete in background
            print(f"  TPU deletion initiated")
        except exceptions.NotFound:
            pass
        except Exception as e:
            print(f"  Delete error: {e}")
    
    def _create_validation_script(self, jax_code: str, pytorch_code: str, 
                                   task_id: str) -> str:
        """Create the validation script to run on TPU."""
        return f'''#!/bin/bash
set -e
exec > /tmp/validation.log 2>&1

echo "Installing dependencies..."
pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -q google-cloud-storage torch numpy

echo "Running validation..."
python3 << 'PYTHON_VALIDATION_EOF'
import json
import time
import numpy as np
import sys

# Task ID for results
TASK_ID = "{task_id}"

results = {{
    "task_id": TASK_ID,
    "compilation": {{"success": False, "error": ""}},
    "correctness": {{"success": False, "error": "", "max_diff": None}},
    "performance": {{"jax_ms": None, "pytorch_ms": None, "speedup": None}},
}}

# ============ JAX CODE ============
JAX_CODE = """
{jax_code.replace('"', '\\"').replace("\\n", "\\\\n")}
"""

# ============ PYTORCH CODE ============
PYTORCH_CODE = """
{pytorch_code.replace('"', '\\"').replace("\\n", "\\\\n")}
"""

def upload_results():
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("{BUCKET_NAME}")
    blob = bucket.blob(f"validation_{{TASK_ID}}.json")
    blob.upload_from_string(json.dumps(results, indent=2))

try:
    # Step 1: Try to compile JAX code
    print("Step 1: Compiling JAX code...")
    jax_namespace = {{"__name__": "__main__"}}
    exec(JAX_CODE.strip(), jax_namespace)
    results["compilation"]["success"] = True
    print("  JAX code compiled successfully")
except Exception as e:
    results["compilation"]["error"] = str(e)[:500]
    print(f"  JAX compilation failed: {{e}}")
    upload_results()
    sys.exit(0)

try:
    # Step 2: Run PyTorch to get reference outputs
    print("Step 2: Running PyTorch reference...")
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    pt_namespace = {{"__name__": "__main__"}}
    exec(PYTORCH_CODE.strip(), pt_namespace)
    
    pt_model = pt_namespace["Model"]()
    pt_inputs = pt_namespace["get_inputs"]()
    
    with torch.no_grad():
        pt_output = pt_model.forward(*pt_inputs)
    
    if isinstance(pt_output, torch.Tensor):
        pt_output_np = pt_output.detach().cpu().numpy()
    else:
        pt_output_np = np.array(pt_output)
    
    print(f"  PyTorch output shape: {{pt_output_np.shape}}")
    
    # Convert PyTorch inputs to numpy for JAX
    pt_inputs_np = []
    for inp in pt_inputs:
        if isinstance(inp, torch.Tensor):
            pt_inputs_np.append(inp.detach().cpu().numpy())
        else:
            pt_inputs_np.append(inp)
    
except Exception as e:
    results["correctness"]["error"] = f"PyTorch reference failed: {{str(e)[:300]}}"
    print(f"  PyTorch failed: {{e}}")
    upload_results()
    sys.exit(0)

try:
    # Step 3: Run JAX with same inputs
    print("Step 3: Running JAX...")
    import jax
    import jax.numpy as jnp
    
    # Get JAX model and run
    jax_model = jax_namespace["Model"]()
    
    # Convert numpy inputs to JAX arrays
    jax_inputs = [jnp.array(x) if isinstance(x, np.ndarray) else x for x in pt_inputs_np]
    
    jax_output = jax_model.forward(*jax_inputs)
    
    if hasattr(jax_output, 'block_until_ready'):
        jax_output.block_until_ready()
    
    jax_output_np = np.array(jax_output)
    print(f"  JAX output shape: {{jax_output_np.shape}}")
    
    # Step 4: Compare outputs
    print("Step 4: Checking correctness...")
    if pt_output_np.shape != jax_output_np.shape:
        results["correctness"]["error"] = f"Shape mismatch: PT={{pt_output_np.shape}}, JAX={{jax_output_np.shape}}"
    else:
        max_diff = float(np.max(np.abs(pt_output_np - jax_output_np)))
        results["correctness"]["max_diff"] = max_diff
        
        # Use relative tolerance for larger values
        if np.allclose(pt_output_np, jax_output_np, rtol=1e-3, atol=1e-3):
            results["correctness"]["success"] = True
            print(f"  Correctness PASSED (max diff: {{max_diff:.6f}})")
        else:
            results["correctness"]["error"] = f"Values differ (max diff: {{max_diff:.6f}})"
            print(f"  Correctness FAILED (max diff: {{max_diff:.6f}})")

except Exception as e:
    results["correctness"]["error"] = f"JAX execution failed: {{str(e)[:300]}}"
    print(f"  JAX execution failed: {{e}}")
    upload_results()
    sys.exit(0)

try:
    # Step 5: Benchmark performance
    print("Step 5: Benchmarking...")
    
    # Benchmark JAX
    @jax.jit
    def jax_forward(*args):
        return jax_model.forward(*args)
    
    # Warmup
    for _ in range(3):
        out = jax_forward(*jax_inputs)
        out.block_until_ready()
    
    # Benchmark
    num_iters = 50
    start = time.perf_counter()
    for _ in range(num_iters):
        out = jax_forward(*jax_inputs)
        out.block_until_ready()
    jax_time = (time.perf_counter() - start) / num_iters * 1000
    
    results["performance"]["jax_ms"] = round(jax_time, 4)
    print(f"  JAX: {{jax_time:.4f}} ms")
    
    # Benchmark PyTorch (CPU on TPU VM, just for reference)
    with torch.no_grad():
        for _ in range(3):
            _ = pt_model.forward(*pt_inputs)
        
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = pt_model.forward(*pt_inputs)
        pt_time = (time.perf_counter() - start) / num_iters * 1000
    
    results["performance"]["pytorch_ms"] = round(pt_time, 4)
    results["performance"]["speedup"] = round(pt_time / jax_time, 2) if jax_time > 0 else 0
    print(f"  PyTorch (CPU): {{pt_time:.4f}} ms")
    print(f"  Speedup: {{results['performance']['speedup']}}x")

except Exception as e:
    print(f"  Benchmark error: {{e}}")

# Upload results
print("Uploading results...")
upload_results()
print("Done!")
PYTHON_VALIDATION_EOF

echo "Validation complete!"
'''
    
    def validate(self, jax_code: str, pytorch_code: str, task_id: str,
                 timeout: int = 180) -> Dict[str, Any]:
        """
        Validate JAX code on TPU.
        
        Args:
            jax_code: Generated JAX code
            pytorch_code: Original PyTorch code  
            task_id: Unique task identifier
            timeout: Max seconds to wait for results
            
        Returns:
            Validation results dict
        """
        # Upload validation script
        script = self._create_validation_script(jax_code, pytorch_code, task_id)
        script_blob = self.bucket.blob(f"scripts/validate_{task_id}.sh")
        script_blob.upload_from_string(script)
        
        # Delete old results
        try:
            self.bucket.blob(f"validation_{task_id}.json").delete()
        except:
            pass
        
        # Create/update TPU with new startup script
        parent = f"projects/{PROJECT_ID}/locations/{self.zone}"
        node_path = f"{parent}/nodes/{self.tpu_name}"
        
        # Check TPU state
        try:
            node = self.client.get_node(name=node_path)
            if node.state.name != "READY":
                if not self.create_tpu():
                    return {"error": "Failed to create TPU"}
        except exceptions.NotFound:
            if not self.create_tpu():
                return {"error": "Failed to create TPU"}
        
        # Run validation via SSH/startup script approach
        # For simplicity, we'll recreate the TPU with the validation script
        
        # Delete and recreate with new script
        try:
            self.client.delete_node(name=node_path).result(timeout=120)
            time.sleep(5)
        except:
            pass
        
        # Create with validation script
        node = tpu_v2.Node()
        node.accelerator_type = self.tpu_type
        node.runtime_version = self._get_runtime_version()
        node.scheduling_config = tpu_v2.SchedulingConfig()
        node.scheduling_config.preemptible = True
        node.network_config = tpu_v2.NetworkConfig()
        node.network_config.network = "default"
        node.network_config.subnetwork = "default"
        node.network_config.enable_external_ips = True
        node.metadata = {"startup-script": script}
        
        try:
            operation = self.client.create_node(
                parent=parent,
                node_id=self.tpu_name,
                node=node,
            )
            operation.result(timeout=300)
        except Exception as e:
            return {"error": f"TPU creation failed: {str(e)[:200]}"}
        
        # Wait for results
        result_blob = self.bucket.blob(f"validation_{task_id}.json")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if result_blob.exists():
                result_blob.reload()
                content = result_blob.download_as_string()
                return json.loads(content)
            time.sleep(5)
        
        return {"error": "Validation timed out"}


class BatchValidator:
    """Validates multiple JAX codes efficiently using a single TPU."""
    
    def __init__(self, tpu_type: str = "v6e-1", zone: str = ZONE):
        self.tpu_type = tpu_type
        self.zone = zone
        self.tpu_name = "jaxbench-batch"
        self.client = get_tpu_client()
        self.storage = get_storage_client()
        self.bucket = self.storage.bucket(BUCKET_NAME)
    
    def validate_batch(self, tasks: list, timeout_per_task: int = 60) -> list:
        """
        Validate multiple tasks on a single TPU.
        
        Args:
            tasks: List of dicts with 'task_id', 'jax_code', 'pytorch_code'
            timeout_per_task: Timeout per task in seconds
            
        Returns:
            List of validation results
        """
        if not tasks:
            return []
        
        # Create batch validation script
        script = self._create_batch_script(tasks)
        
        # Run on TPU
        results = self._run_on_tpu(script, len(tasks) * timeout_per_task)
        
        return results
    
    def _create_batch_script(self, tasks: list) -> str:
        """Create batch validation script."""
        tasks_json = json.dumps(tasks)
        
        return f'''#!/bin/bash
set -e
exec > /tmp/batch_validation.log 2>&1

echo "Installing dependencies..."
pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -q google-cloud-storage torch numpy

echo "Running batch validation..."
python3 << 'BATCH_PYTHON_EOF'
import json
import time
import numpy as np
import sys
import traceback

TASKS = {tasks_json}
BUCKET_NAME = "{BUCKET_NAME}"

all_results = []

for task in TASKS:
    task_id = task["task_id"]
    jax_code = task["jax_code"]
    pytorch_code = task["pytorch_code"]
    
    print(f"\\n{'='*60}")
    print(f"Validating task: {{task_id}}")
    print(f"{'='*60}")
    
    result = {{
        "task_id": task_id,
        "compilation": {{"success": False, "error": ""}},
        "correctness": {{"success": False, "error": "", "max_diff": None}},
        "performance": {{"jax_ms": None, "pytorch_ms": None, "speedup": None}},
    }}
    
    try:
        # Compile JAX
        print("  Compiling JAX...")
        jax_ns = {{"__name__": "__main__"}}
        exec(jax_code.strip(), jax_ns)
        result["compilation"]["success"] = True
        
        # Run PyTorch reference
        print("  Running PyTorch reference...")
        import torch
        torch.manual_seed(42)
        np.random.seed(42)
        
        pt_ns = {{"__name__": "__main__"}}
        exec(pytorch_code.strip(), pt_ns)
        
        pt_model = pt_ns["Model"]()
        pt_inputs = pt_ns["get_inputs"]()
        
        with torch.no_grad():
            pt_out = pt_model.forward(*pt_inputs)
        
        pt_out_np = pt_out.detach().cpu().numpy() if isinstance(pt_out, torch.Tensor) else np.array(pt_out)
        pt_inputs_np = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in pt_inputs]
        
        # Run JAX
        print("  Running JAX...")
        import jax
        import jax.numpy as jnp
        
        jax_model = jax_ns["Model"]()
        jax_inputs = [jnp.array(x) if isinstance(x, np.ndarray) else x for x in pt_inputs_np]
        jax_out = jax_model.forward(*jax_inputs)
        if hasattr(jax_out, 'block_until_ready'):
            jax_out.block_until_ready()
        jax_out_np = np.array(jax_out)
        
        # Check correctness
        print("  Checking correctness...")
        if pt_out_np.shape != jax_out_np.shape:
            result["correctness"]["error"] = f"Shape: PT={{pt_out_np.shape}} vs JAX={{jax_out_np.shape}}"
        else:
            max_diff = float(np.max(np.abs(pt_out_np - jax_out_np)))
            result["correctness"]["max_diff"] = max_diff
            if np.allclose(pt_out_np, jax_out_np, rtol=1e-3, atol=1e-3):
                result["correctness"]["success"] = True
            else:
                result["correctness"]["error"] = f"Values differ (max_diff={{max_diff:.6f}})"
        
        # Benchmark
        print("  Benchmarking...")
        @jax.jit
        def jax_fwd(*args):
            return jax_model.forward(*args)
        
        for _ in range(3):
            jax_fwd(*jax_inputs).block_until_ready()
        
        start = time.perf_counter()
        for _ in range(50):
            jax_fwd(*jax_inputs).block_until_ready()
        jax_ms = (time.perf_counter() - start) / 50 * 1000
        result["performance"]["jax_ms"] = round(jax_ms, 4)
        
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(50):
                pt_model.forward(*pt_inputs)
            pt_ms = (time.perf_counter() - start) / 50 * 1000
        result["performance"]["pytorch_ms"] = round(pt_ms, 4)
        result["performance"]["speedup"] = round(pt_ms / jax_ms, 2) if jax_ms > 0 else 0
        
        print(f"  Result: compile={{result['compilation']['success']}}, correct={{result['correctness']['success']}}, speedup={{result['performance']['speedup']}}x")
        
    except Exception as e:
        error_msg = traceback.format_exc()[:500]
        if not result["compilation"]["success"]:
            result["compilation"]["error"] = error_msg
        else:
            result["correctness"]["error"] = error_msg
        print(f"  Error: {{str(e)[:100]}}")
    
    all_results.append(result)

# Upload all results
print("\\nUploading results...")
from google.cloud import storage
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("batch_results.json")
blob.upload_from_string(json.dumps(all_results, indent=2))
print("Done!")
BATCH_PYTHON_EOF

echo "Batch validation complete!"
'''
    
    def _run_on_tpu(self, script: str, timeout: int) -> list:
        """Run script on TPU and return results."""
        parent = f"projects/{PROJECT_ID}/locations/{self.zone}"
        node_path = f"{parent}/nodes/{self.tpu_name}"
        
        # Delete existing TPU
        try:
            self.client.delete_node(name=node_path).result(timeout=120)
            time.sleep(5)
        except:
            pass
        
        # Delete old results
        try:
            self.bucket.blob("batch_results.json").delete()
        except:
            pass
        
        # Create TPU with script
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
        node.metadata = {"startup-script": script}
        
        print(f"  Creating TPU {self.tpu_type}...")
        try:
            op = self.client.create_node(parent=parent, node_id=self.tpu_name, node=node)
            op.result(timeout=300)
        except Exception as e:
            return [{"error": f"TPU creation failed: {e}"}]
        
        # Wait for results
        print(f"  Waiting for results (timeout: {timeout}s)...")
        result_blob = self.bucket.blob("batch_results.json")
        start = time.time()
        
        while time.time() - start < timeout:
            if result_blob.exists():
                content = result_blob.download_as_string()
                return json.loads(content)
            time.sleep(10)
            print(f"    [{int(time.time() - start)}s] Waiting...")
        
        return [{"error": "Batch validation timed out"}]
    
    def cleanup(self):
        """Delete TPU."""
        node_path = f"projects/{PROJECT_ID}/locations/{self.zone}/nodes/{self.tpu_name}"
        try:
            self.client.delete_node(name=node_path)
            print("  TPU deletion initiated")
        except:
            pass


if __name__ == "__main__":
    print("Testing validator...")
    validator = BatchValidator()
    print("Validator ready")

