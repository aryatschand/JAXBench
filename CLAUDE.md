# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXBench is an LLM-powered benchmark suite that automatically translates PyTorch operators from KernelBench to JAX and validates them on Google Cloud TPUs. It demonstrates JAX's performance advantages over PyTorch/XLA on the same TPU hardware.

## Common Commands

### Running Benchmarks
```bash
# Run first N tasks for a level
python scripts/run_benchmark.py --level 1 --tasks 10 --keep-tpu

# Run all tasks for a level
python scripts/run_benchmark.py --level 1 --all --keep-tpu
python scripts/run_benchmark.py --level 2 --all --keep-tpu

# Run specific tasks
python scripts/run_benchmark.py --level 2 --task-ids "1,5,10" --keep-tpu

# Specify LLM provider/model
python scripts/run_benchmark.py --level 1 --all --provider bedrock --model opus
```

### Testing
```bash
# Run all tests
python tests/run_all_tests.py

# Quick mode (skip slow TPU benchmarks)
python tests/run_all_tests.py --quick

# Individual tests
python tests/test_tpu_connection.py
python tests/test_matmul_benchmark.py
python tests/test_llm_client.py
```

### Visualization
```bash
python scripts/visualize_results.py results/jaxbench_bedrock_*.json
```

## Architecture

### 4-Stage Translation Pipeline

1. **LLM Translation**: PyTorch `nn.Module` → JAX using Bedrock (Claude) or Gemini
2. **Compilation Check**: JAX JIT compilation on TPU; errors fed back for retry (up to 3 attempts)
3. **Correctness Validation**: Compare PyTorch CPU vs JAX TPU outputs using deterministic seeding + weight transfer
4. **Performance Benchmarking**: JAX vs PyTorch/XLA timing on same TPU (5 warmup, 50 benchmark iterations)

### Core Modules

| Module | Purpose |
|--------|---------|
| `src/llm_client.py` | Multi-provider LLM client (Bedrock, Gemini) |
| `src/translator.py` | PyTorch → JAX translation with refinement loop |
| `src/tpu_manager.py` | TPU VM lifecycle management via Google Cloud API |
| `src/validator.py` | TPU validation orchestration |
| `src/pipeline.py` | Pipeline orchestration, task loading, result tracking |
| `scripts/run_benchmark.py` | Main CLI entry point with SSH execution |

### Key Data Flows

- **Input**: KernelBench PyTorch workloads from `KernelBench/KernelBench/level{1,2}/`
- **Output**: Translated JAX code saved to `jaxbench/level{1,2}/`
- **Results**: JSON files in `results/`, progress checkpoints in `results/checkpoint_level*.json`
- **Cache**: Successful translations cached in `.cache/` to avoid re-work

## Key Patterns

### Translation Rules (in `src/translator.py`)
- PyTorch uses **NCHW**, JAX typically uses **NHWC** - code handles conversion
- Generated JAX models include `set_weights()` method for PyTorch weight transfer
- Weight tensors transposed to match JAX's expected layout

### Correctness Validation
- Deterministic seeding: `torch.manual_seed(42)` for identical inputs
- PyTorch reference runs on CPU, JAX runs on TPU with transferred weights
- Tolerance: `rtol=5e-2, atol=0.5` (accounts for TPU bfloat16 precision)

### Performance Benchmarking
- JAX: `output.block_until_ready()` for synchronization
- PyTorch/XLA: `xm.wait_device_ops()` for synchronization (critical - without this, timing is wrong)

### LLM Strategy
- Uses faster/cheaper Sonnet for initial translation
- Falls back to Opus for retries on failure

## Infrastructure

- **TPU Types**: v6e-1, v5e-8, v5e-4, v4-8, v4-16, v4-32 (preemptible by default)
- **GCP**: Project `jaxbench`, zone `us-central1-b`, credentials in `credentials.json`
- **SSH**: Key at `~/.ssh/id_rsa_tpu`, user `REDACTED_SSH_USER`
- **TPU Packages**: jax 0.6.2, torch 2.9.0+cpu, torch_xla 2.9.0, libtpu 0.0.17
