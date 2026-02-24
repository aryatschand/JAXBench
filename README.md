# JAXBench

A benchmark suite for automatically translating PyTorch operators from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) to JAX and validating them on Google Cloud TPUs.

## Overview

JAXBench provides:
1. **Automatic Translation**: Uses LLMs (Claude Opus/Sonnet via AWS Bedrock, or Gemini) to translate PyTorch code to JAX
2. **Rigorous Validation**: Ensures functional correctness by comparing outputs with identical inputs and weights
3. **TPU Performance Benchmarking**: Measures JAX vs PyTorch/XLA performance on the same TPU hardware
4. **Pallas-Ready**: Generated JAX code can be enhanced with custom Pallas kernels for further optimization

## Benchmark Levels

| Level | Description | Tasks | Complexity |
|-------|-------------|-------|------------|
| Level 1 | Single operators (matmul, elementwise) | 100 | Simple |
| Level 2 | Fused operators (Conv+ReLU, Conv+BN+GELU) | 100 | Medium |

---

## Translation & Validation Pipeline

The core of JAXBench is a rigorous 4-stage pipeline that ensures generated JAX code is not only syntactically correct but functionally equivalent and performant.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        JAXBench Translation Pipeline                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   STAGE 1    │     │   STAGE 2    │     │   STAGE 3    │     │   STAGE 4    │
│  Translation │────▶│  Compilation │────▶│  Correctness │────▶│  Performance │
│              │     │    Check     │     │    Check     │     │  Benchmark   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
  │   LLM   │         │   JAX   │         │ PyTorch │         │  JAX    │
  │ (Claude │         │  JIT    │         │   vs    │         │   vs    │
  │  /Gemini│         │ Compile │         │  JAX    │         │PyTorch/ │
  │    )    │         │ on TPU  │         │ Output  │         │  XLA    │
  └─────────┘         └─────────┘         └─────────┘         └─────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │  Same Inputs & Weights │
                                    │  (Deterministic Seed)  │
                                    └───────────────────────┘
```

### Stage 1: LLM Translation (PyTorch → JAX)

The pipeline reads PyTorch `nn.Module` code from KernelBench and uses an LLM to generate equivalent JAX code.

**Input (PyTorch):**
```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    
    def forward(self, x):
        return torch.relu(self.conv(x))
```

**Output (JAX):**
```python
class Model:
    def __init__(self):
        self.weight = jnp.zeros((64, 3, 3, 3))  # OIHW format
        self.bias = jnp.zeros((64,))
    
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
    
    def forward(self, x):
        # Convert NCHW → NHWC for JAX
        x = jnp.transpose(x, (0, 2, 3, 1))
        out = jax.lax.conv_general_dilated(x, self.weight, ...)
        out = jnp.transpose(out, (0, 3, 1, 2))  # Back to NCHW
        return jax.nn.relu(out + self.bias)
```

**Key Translation Rules:**
- PyTorch uses **NCHW** (batch, channels, height, width), JAX typically uses **NHWC**
- Weight tensors must be transposed to match JAX's expected layout
- The LLM is prompted with explicit rules for `Conv2d`, `ConvTranspose2d`, `Conv3d`, etc.
- A `set_weights()` method is generated to accept PyTorch weights

### Stage 2: Compilation Check

The generated JAX code is executed on the TPU to verify it compiles without errors.

```python
# On TPU VM
import jax
import jax.numpy as jnp

# Load generated code
exec(jax_code)
model = Model()

# Create test input
x = jnp.ones((batch, channels, height, width))

# JIT compile and run - this catches syntax/type errors
jit_forward = jax.jit(model.forward)
output = jit_forward(x)
```

**What this catches:**
- Syntax errors in generated code
- Type mismatches (wrong dtypes)
- Shape mismatches in operations
- Invalid JAX API usage

If compilation fails, the error message is fed back to the LLM for up to 3 retry attempts.

### Stage 3: Correctness Validation

This is the most critical stage. We must ensure the JAX code produces **identical outputs** to PyTorch for the same inputs.

#### The Challenge: Randomness

KernelBench tasks use random inputs:
```python
def get_inputs():
    return [torch.randn(batch, channels, height, width)]  # Random!
```

If PyTorch and JAX use different random inputs, we can't compare outputs.

#### The Solution: Deterministic Seeding + Weight Transfer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Correctness Validation Flow                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │  Set Seed = 42  │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Generate Inputs │  torch.manual_seed(42)
    │   (PyTorch)     │  inputs = get_inputs()
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Run PyTorch     │  pt_model = Model()
    │ (CPU Reference) │  pt_output = pt_model(*inputs)
    └────────┬────────┘
             │
    ┌────────▼─────────────────┐
    │ Extract PyTorch Weights  │  weights = {name: param.numpy() 
    │                          │             for name, param in 
    │                          │             pt_model.named_parameters()}
    └────────┬─────────────────┘
             │
    ┌────────▼────────┐
    │ Create JAX Model│  jax_model = Model()
    │ + Set Weights   │  jax_model.set_weights(weights)  ◀── Same weights!
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Convert Inputs  │  jax_inputs = [jnp.array(x.numpy()) 
    │ to JAX Arrays   │               for x in inputs]  ◀── Same inputs!
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Run JAX on TPU  │  jax_output = jax_model.forward(*jax_inputs)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Compare Outputs │  np.allclose(pt_output, jax_output,
    │                 │              rtol=5e-2, atol=0.5)
    └─────────────────┘
```

**Key Implementation Details:**

1. **Same Random Seed**: `torch.manual_seed(42)` ensures identical input tensors
2. **Weight Transfer**: PyTorch weights are extracted via `named_parameters()` and injected into JAX via `set_weights()`
3. **PyTorch on CPU**: Reference output computed on CPU (not TPU) for determinism
4. **JAX on TPU**: JAX code runs on TPU with transferred weights
5. **Tolerance**: We use `rtol=5e-2, atol=0.5` to account for floating-point differences between CPU and TPU

**Why not exact equality?**
- TPU uses bfloat16 internally for some operations
- Different matrix multiplication algorithms (e.g., Strassen vs standard)
- Floating-point associativity differences: `(a+b)+c ≠ a+(b+c)` in FP32

### Stage 4: Performance Benchmarking

After correctness is verified, we benchmark both implementations **on the same TPU**.

#### PyTorch on TPU: PyTorch/XLA

PyTorch doesn't natively support TPUs. We use **PyTorch/XLA** (`torch_xla`), which:
1. Traces PyTorch operations into an XLA HLO graph
2. Compiles the graph for TPU using the XLA compiler
3. Executes on TPU hardware

```python
import torch_xla.core.xla_model as xm

dev = xm.xla_device()  # Get TPU device
model = model.to(dev)  # Move model to TPU
inputs = [x.to(dev) for x in inputs]  # Move inputs to TPU

# Warmup
for _ in range(5):
    output = model(*inputs)
    xm.mark_step()      # Flush XLA graph
    xm.wait_device_ops()  # Synchronize

# Benchmark
times = []
for _ in range(50):
    t0 = time.perf_counter()
    output = model(*inputs)
    xm.mark_step()
    xm.wait_device_ops()  # CRITICAL: Wait for TPU to finish
    times.append(time.perf_counter() - t0)
```

**Critical: `xm.wait_device_ops()`**

Without this synchronization call, PyTorch/XLA returns immediately after *launching* the operation, not after it completes. This would give artificially fast times.

#### JAX on TPU

JAX is designed for XLA from the ground up:

```python
import jax

jit_forward = jax.jit(model.forward)

# Warmup (triggers compilation)
for _ in range(5):
    output = jit_forward(*inputs)
    output.block_until_ready()

# Benchmark
times = []
for _ in range(50):
    t0 = time.perf_counter()
    output = jit_forward(*inputs)
    output.block_until_ready()  # Wait for TPU
    times.append(time.perf_counter() - t0)
```

### Why JAX Can Be Faster Than PyTorch/XLA

Both JAX and PyTorch/XLA ultimately compile to XLA and run on the same TPU hardware. So why can JAX be faster?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Compiler Stack Comparison                                │
└─────────────────────────────────────────────────────────────────────────────┘

    PyTorch/XLA Path:                    JAX Path:
    ─────────────────                    ─────────────
    
    PyTorch Python API                   JAX Python API
           │                                   │
           ▼                                   ▼
    PyTorch Eager Ops                    JAX Primitives
           │                                   │
           ▼                                   │
    torch_xla Tracing ◀── Overhead            │
           │                                   │
           ▼                                   ▼
    XLA HLO Graph ◀───────────────────▶ XLA HLO Graph
           │                                   │
           ▼                                   ▼
    XLA Compiler                         XLA Compiler
           │                                   │
           ▼                                   ▼
    TPU Execution                        TPU Execution
```

**Key Differences:**

1. **Native XLA Design**: JAX was built from scratch for XLA. Its primitives map directly to XLA operations with no translation layer.

2. **Tracing Overhead**: PyTorch/XLA must trace PyTorch's eager operations and convert them to XLA. This tracing can miss optimization opportunities.

3. **Graph Capture**: JAX's `jax.jit` captures the entire computation graph cleanly. PyTorch/XLA's `xm.mark_step()` boundaries can fragment the graph.

4. **Operator Fusion**: JAX's functional style makes it easier for XLA to fuse operations. PyTorch's mutable tensors can inhibit fusion.

5. **Memory Layout**: JAX code can be written to use TPU-optimal layouts (NHWC) directly, while PyTorch/XLA must handle NCHW→NHWC conversion.

**Typical Speedups Observed:**
- Simple matmuls: 1.0-1.3x (both well-optimized)
- Elementwise ops: 1.0x (trivial for both)
- Complex fused ops: 0.9-1.1x (depends on fusion success)
- Custom Pallas kernels: 2-10x (hand-optimized)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on first 10 Level 1 tasks
python scripts/run_benchmark.py --level 1 --tasks 10 --keep-tpu

# Run ALL Level 1 tasks (100 tasks)
python scripts/run_benchmark.py --level 1 --all --keep-tpu

# Run ALL Level 2 tasks (100 tasks)
python scripts/run_benchmark.py --level 2 --all --keep-tpu

# Run specific tasks
python scripts/run_benchmark.py --level 2 --task-ids "1,5,10" --keep-tpu
```

## Command Line Options

```bash
python scripts/run_benchmark.py [OPTIONS]

Options:
  --level N          KernelBench level: 1, 2, or 3 (default: 1)
  --tasks N          Number of tasks to process (default: 10)
  --all              Run all tasks for the level
  --task-ids IDS     Specific task IDs, comma-separated (e.g., "1,5,10")
  --provider         LLM provider: bedrock or gemini (default: bedrock)
  --model            Model for initial translation (default: sonnet)
  --retry-model      Model for retries (default: opus)
  --tpu              TPU type (default: v6e-1)
  --keep-tpu         Keep TPU running after completion
  --max-retries N    Max retry attempts per task (default: 3)
  --no-cache         Disable translation cache (re-translate everything)
```

## Setup

### 1. GCP Credentials

Place your GCP service account JSON at `credentials.json` (or set `GCP_CREDENTIALS_FILE` env var). Required roles:
- TPU Admin
- Compute Admin  
- Storage Admin
- Service Account User

### 2. SSH Key Setup

```bash
# Generate SSH key for TPU access
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_tpu -N ""

# The pipeline automatically configures OS Login on first run
```

### 3. AWS Bedrock (for Claude)

```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-2"
```

### 4. Gemini (optional)

```bash
export GEMINI_API_KEY="your-api-key"
# Then use: --provider gemini --model gemini-3-pro
```

## Results

Results are saved to:
- `jaxbench/level1/` - Translated JAX code for Level 1
- `jaxbench/level2/` - Translated JAX code for Level 2
- `results/jaxbench_<provider>_<timestamp>.json` - Detailed benchmark results
- `results/checkpoint_level<N>.json` - Progress checkpoints
- `.cache/` - Cached successful translations

### Visualizing Results

```bash
# Generate histograms of speedups and correctness
python scripts/visualize_results.py results/jaxbench_bedrock_*.json
```

## Example Results

### Level 1 (Simple Operators)

| Task | Operation | JAX (ms) | PyTorch/XLA (ms) | Speedup | Max Diff |
|------|-----------|----------|------------------|---------|----------|
| 1 | Square matmul (4096×4096) | 0.28 | 0.37 | 1.32× | 0.24 |
| 2 | Rectangular matmul | 0.28 | 0.36 | 1.26× | 0.33 |
| 3 | Batched matmul | 1.68 | 1.68 | 1.0× | 0.14 |
| 4 | Matrix-vector | 7.73 | 7.76 | 1.0× | 0.0 |
| 5 | Scalar multiply | 7.33 | 7.32 | 1.0× | 0.0 |

### Level 2 (Fused Operators)

| Task | Operation | JAX (ms) | PyTorch/XLA (ms) | Speedup | Max Diff |
|------|-----------|----------|------------------|---------|----------|
| 1 | Conv2D+ReLU+BiasAdd | 2.18 | 2.33 | 1.07× | 0.02 |
| 4 | Conv2D+Mish+Mish | 5.61 | 5.91 | 0.95× | 0.15 |
| 5 | ConvTranspose2D+Subtract+Tanh | 11.14 | 11.14 | 1.0× | 0.08 |

## Project Structure

```
JAXBench/
├── scripts/
│   ├── run_benchmark.py      # Main benchmark pipeline script
│   └── visualize_results.py  # Results visualization
├── src/
│   ├── llm_client.py         # LLM client (Bedrock/Gemini)
│   ├── tpu_manager.py        # TPU VM lifecycle management
│   ├── translator.py         # PyTorch → JAX translation
│   ├── validator.py          # TPU validation logic
│   └── pipeline.py           # Pipeline orchestration
├── tests/
│   ├── run_all_tests.py           # Run all tests
│   ├── test_tpu_connection.py     # TPU connectivity tests
│   ├── test_matmul_benchmark.py   # JAX vs PyTorch/XLA benchmark test
│   ├── test_llm_client.py         # LLM API tests
│   ├── test_manual_kernels.py     # Manual kernel testing utility
│   └── test_tpu.py                # TPU allocation utility
├── jaxbench/
│   ├── level1/               # Translated JAX workloads (Level 1)
│   └── level2/               # Translated JAX workloads (Level 2)
├── KernelBench/              # Cloned PyTorch reference
├── results/                  # Benchmark results (JSON)
├── logs/                     # Execution logs
├── .cache/                   # Translation cache
└── requirements.txt
```

## Testing

Run tests to verify your setup is correct:

```bash
# Run all tests
python tests/run_all_tests.py

# Quick tests (skip TPU benchmark)
python tests/run_all_tests.py --quick

# Test TPU connection only
python tests/test_tpu_connection.py

# Test matmul benchmark (JAX + PyTorch/XLA)
python tests/test_matmul_benchmark.py

# Test LLM clients
python tests/test_llm_client.py
```

## Extending with Pallas

The generated JAX code can be enhanced with custom [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) kernels for TPU optimization:

```python
from jax.experimental import pallas as pl

# Replace jnp.matmul with custom Pallas kernel
def pallas_matmul(a, b):
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=(m // block_m, n // block_n),
        ...
    )(a, b)
```

Pallas allows you to write custom TPU kernels that can significantly outperform auto-generated XLA code by:
- Explicit memory tiling for TPU's HBM/VMEM hierarchy
- Custom data layouts optimized for TPU's systolic array
- Fusing operations that XLA doesn't automatically fuse

## TPU Configuration

Working package versions on TPU v6e:

| Package | Version |
|---------|---------|
| libtpu | 0.0.17 |
| jax | 0.6.2 |
| torch | 2.9.0+cpu |
| torch_xla | 2.9.0 |

## Troubleshooting

### TPU Preempted
Preemptible TPUs may be reclaimed. The pipeline automatically handles this by deleting and recreating the TPU.

### Package Version Mismatch
The pipeline installs specific versions:
- `jax[tpu]` from Google's libtpu releases
- `torch==2.9.0+cpu` (CPU-only)
- `torch_xla[tpu]` matching torch version

### SSH Connection Issues
```bash
# Manual SSH to TPU
ssh -i ~/.ssh/id_rsa_tpu sa_<SERVICE_ACCOUNT_ID>@<TPU_IP>
```

## License

MIT
