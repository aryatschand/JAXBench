# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXBench is an LLM-powered benchmark suite for TPU kernel optimization:
1. **Translation**: Automatically translates PyTorch operators from KernelBench to JAX using LLMs (Claude via AWS Bedrock, or Gemini)
2. **Evaluation**: Validates correctness and benchmarks performance on Google Cloud TPUs
3. **Optimization**: Generates optimized Pallas TPU kernels using LLMs

## Common Commands

### Setup
```bash
pip install -r requirements.txt
# TPU packages (jax[tpu], torch, torch_xla) are installed on TPU VMs automatically
```

### Translation (PyTorch → JAX)
```bash
python -m torch_to_jax.run --level 1 --tasks 10 --keep-tpu
python -m torch_to_jax.run --level 1 --all --keep-tpu
python -m torch_to_jax.run --level 1 --all --provider bedrock --model opus
python -m torch_to_jax.run --level 2 --task-ids "1,5,10" --keep-tpu
# Key flags: --level N, --tasks N, --all, --task-ids, --provider (bedrock|gemini),
#   --model (sonnet|opus|haiku), --tpu (v6e-1|v5e-8|etc), --keep-tpu, --max-retries N, --no-cache
```

### Pallas Optimization
```bash
python -m pallas_optimization.run --list              # List available workloads
python -m pallas_optimization.run --workload llama3_8b_gqa
```

### Tests
```bash
python tests/run_all_tests.py             # All tests
python tests/run_all_tests.py --quick     # Skip slow TPU benchmark
python tests/run_all_tests.py --tpu-only  # TPU tests only
python tests/run_all_tests.py --llm-only  # LLM client tests only
python tests/test_tpu_connection.py       # Single test file
python tests/test_llm_client.py           # Test LLM API access
```

Note: Tests use direct Python execution (no pytest). Each test file has a `main()` entry point.

### Evaluation API
```python
from evaluation.evaluate_kernel import evaluate_kernel, HardwareConfig

result = evaluate_kernel(
    generated_code=code,                         # LLM-generated Pallas code
    benchmark_ref="real_workloads:llama3:gqa",   # Benchmark reference
    hardware_config=HardwareConfig(tpu_type="v5e-4"),
)
# Returns: EvaluationResult with correct, speedup, timing, etc.
```

## Architecture

### 4-Stage Translation Pipeline

```
Stage 1: LLM Translation    → Generate JAX code from PyTorch (Sonnet first, Opus for retries)
Stage 2: Compilation Check   → Verify JAX code compiles on TPU via JIT
Stage 3: Correctness Check   → Compare outputs with identical inputs/weights (seed=42)
Stage 4: Performance Bench   → Measure JAX vs PyTorch/XLA timing on TPU
```

Key design: TPU is allocated once and reused for all validations. Successful translations are cached in `.cache/` to avoid re-work on restart. Progress is saved incrementally to `results/checkpoints/`.

### Core Modules

| Module | Purpose |
|--------|---------|
| `torch_to_jax/run.py` | Main translation pipeline CLI (orchestrates all 4 stages) |
| `torch_to_jax/translator.py` | LLM-based PyTorch→JAX translator |
| `torch_to_jax/llm_client.py` | Multi-provider LLM client (BedrockClient, GeminiClient) |
| `evaluation/evaluate_kernel.py` | Main API for kernel evaluation in optimization loops (partially skeleton) |
| `evaluation/validator.py` | Correctness validation on TPU |
| `evaluation/benchmarker.py` | Performance timing with proper TPU synchronization |
| `evaluation/workload_registry.py` | Workload definitions (register_workload/get_workload pattern) |
| `pallas_optimization/translator.py` | JAX → Pallas kernel translator |
| `pallas_optimization/prompts.py` | Extensive Pallas-specific LLM prompts (operation detection, strategies) |
| `infrastructure/tpu_manager.py` | TPU VM lifecycle (create/delete/SSH), handles preemption automatically |

### Benchmark References

Format: `type:category:identifier`
- `kernelbench:level1:1` — Task 1 from KernelBench level 1 (original PyTorch)
- `jaxkernelbench:level2:5` — JAX translation of task 5, level 2
- `real_workloads:llama3:gqa` — Llama3 GQA attention
- `real_workloads:llama3:rope` — Llama3 RoPE
- `real_workloads:gemma3:sliding` — Gemma3 sliding window attention

### Workload Registry

New workloads are added by registering a `WorkloadConfig` in `evaluation/workload_registry.py`:
```python
register_workload(WorkloadConfig(
    name="model_op", model="llama3", category="attention",
    config={...}, input_generator=fn, baseline_fn=fn,
    rtol=1e-2, atol=1e-2,
))
```

Currently registered: llama3 (gqa, rope, swiglu), gemma3 (sliding window).

## Key Patterns

### Translation Rules
- PyTorch uses **NCHW**, JAX uses **NHWC** — conversion handled automatically
- Generated JAX models include `set_weights()` method for weight transfer from PyTorch
- Weight tensors transposed to match JAX's expected layout
- Correctness tolerance: `rtol=5e-2, atol=0.5` (accounts for CPU vs TPU float differences)

### TPU Synchronization (Critical for Benchmarking)
- JAX: `output.block_until_ready()` — waits for actual TPU execution
- PyTorch/XLA: `xm.mark_step()` + `xm.wait_device_ops()` — flushes and waits
- Without these, you measure launch time, not execution time

### LLM Retry Strategy
- Initial translation uses Sonnet (faster/cheaper)
- Failed translations retry with Opus (stronger reasoning), up to 3 attempts
- Error messages from compilation/validation are fed back to LLM for context

## Infrastructure

- **TPU Types**: v6e-1, v5e-8, v5e-4, v4-8, v4-16, v4-32 (preemptible)
- **GCP**: Project `jaxbench`, zone `us-central1-b`
- **SSH**: Key at `~/.ssh/id_rsa_tpu`, user set via `TPU_SSH_USER` env var
- **TPU Packages**: jax 0.6.2, torch 2.9.0+cpu, torch_xla 2.9.0
- **Credentials**: GCP service account JSON path set via `GCP_CREDENTIALS_FILE` env var (default: `credentials.json`)
