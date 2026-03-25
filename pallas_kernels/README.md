# Pallas Kernels

Upstream JAX Pallas TPU kernels wrapped as JAXBench workloads for optimization by autocomp.

## Source

All kernel code is copied verbatim from **JAX 0.6.2** (`jax.experimental.pallas.ops.tpu`). The
TPU VM runs JAX 0.6.2, so the kernel code must match that version exactly. Each workload file
contains the unmodified upstream kernel followed by a thin JAXBench wrapper (`CONFIG`,
`create_inputs`, `workload`).

| Workload | Upstream module | Model shape | Pallas time |
|----------|----------------|-------------|------------:|
| `matmul.py` | `pallas.ops.tpu.matmul` | Llama-3.1-70B | 22.7 ms |
| `flash_attention.py` | `pallas.ops.tpu.flash_attention` | Llama-3.1-70B | 6.2 ms |
| `splash_attention.py` | `pallas.ops.tpu.splash_attention` | Llama-3.1-70B | 5.9 ms |
| `paged_attention.py` | `pallas.ops.tpu.paged_attention` | Llama-3.1-70B | 1.6 ms |
| `ragged_paged_attention.py` | `pallas.ops.tpu.ragged_paged_attention` | Llama-3.1-70B | 1.6 ms |
| `megablox_gmm.py` | `pallas.ops.tpu.megablox.gmm` | Qwen3-235B-A22B | 21.3 ms |

Times measured on TPU v6e-1 with bf16, median of 100 trials.

## File structure

Each workload file has two parts:

1. **Kernel code** (top of file) — copied from the installed JAX 0.6.2 package on the TPU VM.
   This is the code that autocomp agents will try to optimize. Do not modify it by hand.

2. **JAXBench wrapper** (bottom of file) — `CONFIG` dict with model shape parameters and
   tolerances, `create_inputs()` to generate deterministic inputs, and `workload()` that
   calls the kernel entry point.

```
pallas_kernels/
├── matmul.py                  # 6 workload files (kernel + wrapper)
├── flash_attention.py
├── splash_attention.py
├── paged_attention.py
├── ragged_paged_attention.py
├── megablox_gmm.py
├── jax_references/            # Pure-JAX reference implementations for correctness checks
│   ├── matmul.py
│   ├── flash_attention.py
│   ├── splash_attention.py
│   ├── paged_attention.py
│   ├── ragged_paged_attention.py
│   └── megablox_gmm.py
├── check_references.py        # Validate Pallas kernels against JAX references on TPU
└── README.md
```

## JAX references

`jax_references/` contains pure-JAX (no Pallas) implementations of each kernel, matching the
upstream JAX test references. These are used by `check_references.py` and by `jaxbench_runner.py`
to verify that optimized kernels produce numerically correct output.

Some references (megablox_gmm, ragged_paged_attention) use data-dependent control flow and
cannot be `jit`-compiled. These set `_skip_jit = True` so the checker runs them eagerly.

## Correctness checking

Upload to the TPU VM and run:

```bash
cd ~/pallas_check/pallas_kernels
python check_references.py
```

Expected output — all 6 should show `PASS (local)`:

```
matmul                         PASS (local)
flash_attention                PASS (local)
megablox_gmm                   PASS (local)
splash_attention               PASS (local)
paged_attention                PASS (local)
ragged_paged_attention         PASS (local)
```

## Tolerances

Each workload's `CONFIG` includes `atol` and `rtol` matching the upstream JAX test tolerances.
These are used by both `check_references.py` and `jaxbench_runner.py` (autocomp's evaluation
harness).

## Updating kernels

If the TPU VM JAX version changes, re-extract the kernel source from the installed package:

```bash
# On the TPU VM
python -c "import jax.experimental.pallas.ops.tpu.flash_attention as m; print(m.__file__)"
```

Then replace the kernel portion of each workload file (everything above `CONFIG =`) with the
new source, keeping the wrapper unchanged.
