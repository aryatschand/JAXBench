# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXBench is a benchmark suite for JAX and TPU kernel optimization. Three benchmark suites at root level, plus a translation pipeline:
1. **jaxkernelbench/**: 200+ LLM-translated PyTorch→JAX operators (from KernelBench)
2. **real_workloads/**: 30 hand-written ops from 6 modern LLM families (from MaxText)
3. **tokamax/**: 12 workloads from openxla/tokamax across 6 operation categories
4. **torch_to_jax/**: LLM-powered translation pipeline (Claude via AWS Bedrock, or Gemini)

Infrastructure code (evaluation/, infrastructure/, tests/, pallas_optimization/) is kept locally but not tracked in git.

## Common Commands

### Setup
```bash
pip install -r requirements.txt
# TPU packages (jax[tpu], torch, torch_xla) are installed on TPU VMs automatically
```

### Run Individual Workloads
```bash
# Each workload file is self-contained, prints JSON
python real_workloads/llama3/llama3_8b_gqa.py
python tokamax/attention/mixtral_8x7b_attention.py
```

### Benchmark Suites on TPU
```bash
python -m real_workloads.run_benchmarks --tpu v6e-1 --keep-tpu
python -m tokamax.run_benchmarks --tpu v6e-1 --keep-tpu
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

### Pallas Optimization (local only)
```bash
python -m pallas_optimization.run --list
python -m pallas_optimization.run --workload llama3_8b_gqa
```

### Tests (local only)
```bash
python tests/run_all_tests.py             # All tests
python tests/run_all_tests.py --quick     # Skip slow TPU benchmark
python tests/test_tpu_connection.py       # Single test file
python tests/test_llm_client.py           # Test LLM API access
```

## Architecture

### Workload File Format

Every workload file (real_workloads and tokamax) follows the same template:
```python
CONFIG = {'name': '...', 'model': '...', 'operator': '...', ...}
def create_inputs(dtype=jnp.bfloat16): ...   # Deterministic seed=42
def workload(*inputs): ...                    # Vanilla JAX baseline
def benchmark(num_warmup=5, num_iters=100): ... # Returns JSON dict
if __name__ == '__main__': print(json.dumps(benchmark()))
```

### Benchmark Suites

| Suite | Workloads | Source | Organization |
|-------|-----------|--------|--------------|
| jaxkernelbench/ | 202 | KernelBench (LLM-translated) | level1/, level2/ |
| real_workloads/ | 30 | MaxText | by model family (llama3/, gemma3/, etc.) |
| tokamax/ | 12 | openxla/tokamax | by operation (attention/, cross_entropy/, etc.) |

### 4-Stage Translation Pipeline

```
Stage 1: LLM Translation    → Generate JAX code from PyTorch (Sonnet first, Opus for retries)
Stage 2: Compilation Check   → Verify JAX code compiles on TPU via JIT
Stage 3: Correctness Check   → Compare outputs with identical inputs/weights (seed=42)
Stage 4: Performance Bench   → Measure JAX vs PyTorch/XLA timing on TPU
```

### Core Modules (git-tracked)

| Module | Purpose |
|--------|---------|
| `torch_to_jax/run.py` | Main translation pipeline CLI |
| `torch_to_jax/translator.py` | LLM-based PyTorch→JAX translator |
| `torch_to_jax/llm_client.py` | Multi-provider LLM client (BedrockClient, GeminiClient) |

### Local-Only Modules (not in git)

| Module | Purpose |
|--------|---------|
| `evaluation/` | Kernel evaluation, correctness validation, benchmarking |
| `infrastructure/tpu_manager.py` | TPU VM lifecycle (create/delete/SSH) |
| `pallas_optimization/` | JAX → Pallas kernel translator + prompts |
| `tests/` | Test suite |

## Key Patterns

### TPU Synchronization (Critical for Benchmarking)
- JAX: `output.block_until_ready()` — waits for actual TPU execution
- PyTorch/XLA: `xm.mark_step()` + `xm.wait_device_ops()` — flushes and waits
- Without these, you measure launch time, not execution time

### Translation Rules
- PyTorch uses **NCHW**, JAX uses **NHWC** — conversion handled automatically
- Generated JAX models include `set_weights()` method for weight transfer
- Correctness tolerance: `rtol=5e-2, atol=0.5`

### LLM Retry Strategy
- Initial translation uses Sonnet (faster/cheaper)
- Failed translations retry with Opus (stronger reasoning), up to 3 attempts
- Error messages from compilation/validation are fed back to LLM

## Infrastructure

- **TPU Types**: v6e-1, v5e-8, v5e-4, v4-8, v4-16, v4-32 (preemptible)
- **GCP**: Project `jaxbench`, zone `us-central1-b`
- **SSH**: Key at `~/.ssh/id_rsa_tpu`, user set via `TPU_SSH_USER` env var
- **TPU Packages**: jax 0.6.2, torch 2.9.0+cpu, torch_xla 2.9.0
- **Credentials**: GCP service account JSON path set via `GCP_CREDENTIALS_FILE` env var (default: `credentials.json`)
