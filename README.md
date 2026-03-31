# JAXBench

JAX and TPU kernel benchmarks. This repository contains two suites:

| Suite | Contents |
|-------|----------|
| [`jaxkernelbench/`](jaxkernelbench/) | 200 LLM-translated PyTorch→JAX operators (KernelBench), levels 1–2 |
| [`priority_kernels/`](priority_kernels/) | Core LLM operator workloads (baseline, optimized, and Pallas variants where applicable) |

Benchmarks target **TPU** (e.g. v6e-1) with recent **JAX** releases. Run individual workloads from their directories; see each suite’s READMEs for layout and `run_benchmarks.py` where provided.

## Setup

```bash
pip install -r requirements.txt
```

## License

MIT
