# Pallas Kernel Optimization Pipeline

This module provides tools for generating, evaluating, and benchmarking Pallas TPU kernels. The goal is to explore whether LLM-generated Pallas kernels can achieve speedups over JAX's native implementations.

## Key Findings (Summary)

From our experiments:

| Approach | Result | Notes |
|----------|--------|-------|
| INT8 quantized matmul | JAX wins | Native TPU INT8 units, 3.2x over FP32 |
| Custom Pallas matmul | 46-110x slower | Overhead from grid/block management |
| Exotic formats (FP E2M1, etc.) | Decode overhead | Table lookups negate compression benefits |

**Key insight**: JAX's XLA compiler is highly optimized for standard operations. Pallas is most useful for operations JAX doesn't support natively.

## Installation

Requires JAX with TPU support:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Usage

### Run Pallas Benchmark
```bash
# Compare JAX vs Pallas for quantized matmul
python pallas/scripts/run_pallas_benchmark.py --size 4096

# Run all exotic quantization formats
python pallas/scripts/run_exotic_benchmark.py --size 8192

# Generate a new Pallas kernel via LLM
python pallas/scripts/generate_kernel.py --task "fused_softmax" --provider bedrock
```

### Evaluate a Kernel on TPU
```bash
# SSH to TPU and run benchmark
python pallas/scripts/evaluate_kernel.py \
    --kernel pallas/kernels/quantization/int5_packed.py \
    --tpu-ip REDACTED_IP
```

## Directory Structure

```
pallas/
├── kernels/
│   ├── quantization/      # Quantized matmul kernels
│   │   ├── int5_packed.py # True 5-bit packing
│   │   ├── int8_fused.py  # INT8 with fused dequant
│   │   └── exotic_formats.py
│   ├── sparsity/          # Sparse kernels
│   └── matmul/            # Base matmul patterns
├── prompts/               # LLM prompts for kernel generation
├── scripts/               # Benchmark and evaluation scripts
├── results/               # Benchmark results (JSON)
└── utils/                 # Helper utilities
```

## Kernel Development Guide

### Writing a Pallas TPU Kernel

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def my_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    """Basic tiled matmul pattern."""
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...],
                           preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)

def matmul_pallas(x, y, bm=128, bk=128, bn=128):
    m, k = x.shape
    _, n = y.shape

    return pl.pallas_call(
        functools.partial(my_kernel, nsteps=k // bk),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
    )(x, y)
```

### Common Pitfalls

1. **Constant capture**: Pallas kernels cannot capture JAX arrays. Pass them as inputs or use Python floats.
   ```python
   # BAD - captures scale as JAX array
   def kernel(x_ref, z_ref):
       z_ref[...] = x_ref[...] * scale  # scale is a JAX array

   # GOOD - use Python float
   scale_float = float(scale)
   def kernel(x_ref, z_ref):
       z_ref[...] = x_ref[...] * scale_float
   ```

2. **Block sizes**: Last two dimensions must be divisible by 8 and 128 respectively.

3. **Dynamic indexing**: Avoid `fori_loop` + `dynamic_slice` - causes 27-332x slowdowns on TPU.

## Benchmark Results

### Quantization Formats (8192x8192)

| Format | Speedup vs FP32 | Accuracy | Notes |
|--------|-----------------|----------|-------|
| INT5 (JAX native) | 3.40x | 0.989 | Uses native INT8 path |
| INT6 (JAX native) | 3.40x | 0.998 | Best accuracy-speed |
| INT8 (JAX native) | 3.21x | 0.999 | Standard quantization |
| FP E2M1 (4-bit) | 0.81x | 0.950 | Table lookup overhead |
| BFP (block float) | 0.29x | 0.968 | Reshape overhead |

### JAX vs Pallas (Same Operation)

| Method | 4096 Time | 8192 Time | vs JAX |
|--------|-----------|-----------|--------|
| JAX INT5 | 0.21 ms | 0.96 ms | 1.00x |
| Pallas INT5 | 13.39 ms | 105.66 ms | 0.01x |

**Conclusion**: Custom Pallas is 46-110x slower than JAX for standard quantized matmul.

## Contributing

When adding new kernels:

1. Create kernel in `pallas/kernels/<category>/`
2. Add benchmark to `pallas/scripts/`
3. Document results in `pallas/results/`
4. Update this README with findings
