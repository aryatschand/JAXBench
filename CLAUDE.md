# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXBench is an LLM-powered benchmark suite for TPU kernel optimization:
1. **Translation**: Automatically translates PyTorch operators from KernelBench to JAX
2. **Evaluation**: Validates correctness and benchmarks performance on Google Cloud TPUs
3. **Optimization**: Generates optimized Pallas TPU kernels using LLMs

## Directory Structure

```
JAXBench/
├── benchmarks/                    # All benchmark workloads
│   ├── kernelbench/              # Original PyTorch (from KernelBench)
│   ├── jaxkernelbench/           # Translated JAX versions
│   └── real_workloads/           # Model-specific JAX baselines (llama3, gemma3, etc.)
│
├── torch_to_jax/                  # PyTorch → JAX translation pipeline
├── evaluation/                    # Evaluation framework + evaluate_kernel.py API
├── pallas_optimization/           # Pallas kernel generation
├── infrastructure/                # TPU/hardware management
├── results/                       # Output + visualization
└── tests/                         # Unit tests
```

## Common Commands

### Translation (PyTorch → JAX)
```bash
# Run translation for a level
python -m torch_to_jax.run --level 1 --tasks 10 --keep-tpu

# Run all tasks
python -m torch_to_jax.run --level 1 --all --keep-tpu

# Specify LLM provider/model
python -m torch_to_jax.run --level 1 --all --provider bedrock --model opus
```

### Evaluation
```bash
# Evaluate a single kernel (main API for optimization loop)
from evaluation.evaluate_kernel import evaluate_kernel, HardwareConfig

result = evaluate_kernel(
    generated_code="def pallas_kernel(...): ...",
    benchmark_ref="real_workloads:llama3:gqa",
    hardware_config=HardwareConfig(tpu_type="v5e-4"),
)
```

### Pallas Optimization
```bash
python -m pallas_optimization.run --workload llama3_8b_gqa
python -m pallas_optimization.run --list
```

## Core Modules

| Module | Purpose |
|--------|---------|
| `torch_to_jax/` | PyTorch → JAX translation pipeline |
| `evaluation/evaluate_kernel.py` | **Main API** for kernel evaluation in optimization loops |
| `evaluation/validator.py` | Correctness checking |
| `evaluation/benchmarker.py` | Performance timing |
| `evaluation/workload_registry.py` | Workload definitions |
| `pallas_optimization/` | Pallas kernel generation |
| `infrastructure/tpu_manager.py` | TPU VM lifecycle management |

## Key Patterns

### Translation Rules
- PyTorch uses **NCHW**, JAX uses **NHWC** - conversion handled automatically
- Generated JAX models include `set_weights()` method for weight transfer
- Weight tensors transposed to match JAX's expected layout

### Evaluation API
```python
# evaluate_kernel.py - main entry point for optimization loops
result = evaluate_kernel(
    generated_code=code,                    # LLM-generated Pallas code
    benchmark_ref="real_workloads:llama3:gqa",  # Benchmark reference
    hardware_config=HardwareConfig(...),    # TPU config
)
# Returns: EvaluationResult with correct, speedup, timing, etc.
```

### Benchmark References
- `kernelbench:level1:1` - Task 1 from KernelBench level 1
- `jaxkernelbench:level2:5` - JAX translation of task 5, level 2
- `real_workloads:llama3:gqa` - Llama3 GQA attention
- `real_workloads:llama3:rope` - Llama3 RoPE
- `real_workloads:gemma3:sliding` - Gemma3 sliding window attention

## Infrastructure

- **TPU Types**: v6e-1, v5e-8, v5e-4, v4-8, v4-16, v4-32 (preemptible)
- **GCP**: Project `jaxbench`, zone `us-central1-b`
- **SSH**: Key at `~/.ssh/id_rsa_tpu`, user `REDACTED_SSH_USER`
- **TPU Packages**: jax 0.6.2, torch 2.9.0+cpu, torch_xla 2.9.0
