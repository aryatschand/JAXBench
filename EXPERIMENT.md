# Pallas Kernel Generation Experiment

Automated translation of JAX workloads to Pallas TPU kernels using frontier LLMs, evaluated on Google Cloud TPU v6e.

## Setup

### Models
- **GPT-5.3** (`gpt-5.3-chat-latest`) — OpenAI, 1-shot generation
- **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`) — Google, 1-shot generation

### Hardware
- **TPU v6e** on Google Cloud (`ch-llm` project, `us-east5-a`)
- **JAX 0.6.2** with Pallas Mosaic backend

### Workloads
- **jaxkernelbench Level 1** — 100 simple single-op workloads (matmul, activations, norms, convolutions)
- **jaxkernelbench Level 2** — 100 complex multi-op workloads (fused conv+norm+activation chains)
- **Priority Kernels** — 17 production-grade patterns (attention variants, MoE, MLP, etc.)
- **8 workloads excluded** — their baseline JAX code fails on TPU (see [Baseline Validation](#baseline-validation))
- **209 valid workloads** used for evaluation

### Methodology
1. Each LLM receives a system prompt with detailed TPU Pallas API rules and a concrete working example
2. The LLM sees the original JAX workload and must rewrite it using Pallas kernels while preserving the exact interface
3. Generated code is deployed to the TPU and evaluated for:
   - **Correctness**: `jnp.allclose(original_output, pallas_output, atol=1e-2, rtol=1e-2)`
   - **Performance**: median over 20 timed iterations after 3 warmup runs
   - **Speedup**: `original_time / pallas_time`

---

## Results Summary

### Overall (209 valid workloads)

| Model | Correct | Rate | Faster than JAX | Wrong | Errors |
|-------|---------|------|-----------------|-------|--------|
| GPT-5.3 | 67 | **32.1%** | 2 | 10 | 132 |
| Gemini 3.1 Pro | 39 | 18.7% | **6** | 0 | 170 |
| Best-of-2 | **89** | **42.6%** | 8 | — | — |

### By Suite

| Suite | GPT-5.3 | Gemini 3.1 Pro | Best-of-2 |
|-------|---------|----------------|-----------|
| Level 1 (95 valid) | 18 (18.9%) | 22 (23.2%) | 33 (34.7%) |
| Level 2 (99 valid) | **45 (45.5%)** | 16 (16.2%) | **52 (52.5%)** |
| Priority (15 valid) | 4 (26.7%) | 1 (6.7%) | 5 (33.3%) |

### Speedup Distribution (correct kernels only)

| Metric | GPT-5.3 (67 kernels) | Gemini 3.1 Pro (39 kernels) |
|--------|---------------------|----------------------------|
| Min | 0.01x | 0.03x |
| Median | 0.20x | 0.38x |
| Mean | 0.36x | **0.63x** |
| Max | 3.48x | **3.87x** |

---

## Kernels Faster than JAX

### GPT-5.3

| Kernel | Speedup | Original | Pallas |
|--------|---------|----------|--------|
| `14_Gemm_Divide_Sum_Scaling` | **3.48x** | 1.12ms | 0.32ms |
| `flex_attention` | 1.01x | 2.93ms | 2.90ms |

### Gemini 3.1 Pro

| Kernel | Speedup | Original | Pallas |
|--------|---------|----------|--------|
| `95_CrossEntropyLoss` | **3.87x** | 2.06ms | 0.53ms |
| `97_ScaledDotProductAttention` | 1.36x | 10.36ms | 7.64ms |
| `38_L1Norm_` | 1.31x | 0.40ms | 0.30ms |
| `39_L2Norm_` | 1.31x | 0.39ms | 0.30ms |
| `42_Max_Pooling_2D` | 1.27x | 18.21ms | 14.38ms |
| `28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply` | 1.10x | 0.17ms | 0.16ms |

---

## Priority Kernels Detail

| Kernel | GPT-5.3 | Gemini 3.1 Pro |
|--------|---------|----------------|
| `flex_attention` | 1.01x | error |
| `mla_attention` | 0.58x | error |
| `rms_norm` | error | 0.72x |
| `gemm` | 0.19x | error |
| `ragged_dot` | 0.17x | error |
| `cross_entropy` | error | error |
| `flash_attention` | error | error |
| `gqa_attention` | error | error |
| `mamba2_ssd` | error | error |
| `paged_attention` | error | error |
| `retnet_retention` | error | error |
| `sparse_attention` | error | error |
| `sparse_moe` | error | error |
| `swiglu_mlp` | error | error |
| `triangle_multiplication` | error | error |
| `megablox_gmm` | *baseline broken* | *baseline broken* |
| `ragged_paged_attention` | *baseline broken* | *baseline broken* |

---

## Model Complementarity

The two models solve largely different workloads:

| Category | Count |
|----------|-------|
| Both correct | 17 |
| GPT-5.3 only | 50 |
| Gemini only | 22 |
| Neither | 120 |
| **Union (best-of-2)** | **89 / 209 (42.6%)** |

Taking the best result from either model yields 42.6% correctness — a 33% improvement over GPT alone and 128% over Gemini alone.

---

## Error Analysis

### GPT-5.3 (132 errors)

| Category | Count | Description |
|----------|-------|-------------|
| Other | 44 | Various Pallas API misuse |
| Block alignment | 34 | Block shape dimensions not meeting TPU `(8, 128)` requirements |
| OOM | 22 | Block sizes too large for TPU VMEM / HBM |
| Traced boolean | 11 | Python `if/else` on JAX traced values |
| Unimplemented op | 10 | Pallas TPU doesn't support the operation yet |
| Timeout | 4 | Execution exceeded 300s |
| TPU init (transient) | 3 | Device busy between evaluations |
| Spec mismatch | 3 | `in_specs`/`out_specs` don't match inputs |
| Missing attribute | 1 | Generated code missing required class attribute |

### Gemini 3.1 Pro (170 errors)

| Category | Count | Description |
|----------|-------|-------------|
| Other | 123 | Various Pallas API misuse |
| Missing attribute | 12 | Generated code missing `Model` class or required methods |
| OOM | 12 | Block sizes too large for TPU memory |
| Syntax error | 9 | Unterminated strings, invalid Python |
| Unimplemented op | 6 | Unsupported Pallas TPU primitives |
| Block alignment | 5 | Block shape dimension requirements |
| Traced boolean | 3 | Python control flow on traced values |

---

## Baseline Validation

All 217 JAX workloads were validated on the TPU to ensure the baselines themselves run correctly. **209/217 passed**, with 8 failures:

| Suite | Workload | Error |
|-------|----------|-------|
| Level 1 | `50_conv_standard_2D__square_input__square_kernel` | `NoneType` in transpose |
| Level 1 | `55_conv_standard_2D__asymmetric_input__square_kernel` | `NoneType.shape` |
| Level 1 | `58_conv_transposed_3D__asymmetric_input__asymmetric_kernel` | Timeout/hang |
| Level 1 | `68_conv_transposed_3D__square_input__asymmetric_kernel` | Timeout/hang |
| Level 1 | `78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__` | Conv dimension mismatch |
| Level 2 | `22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish` | Missing `matmul_weight` attribute |
| Priority | `megablox_gmm` | Dynamic slice indices (traced) |
| Priority | `ragged_paged_attention` | Traced int `__index__()` |

These 8 workloads are excluded from all reported metrics.

---

## System Prompt

The system prompt used for both models includes:
- Explicit `PrefetchScalarGridSpec` constructor signature with `num_scalar_prefetch=0` as required first argument
- `BlockSpec` rules for `in_specs` / `out_specs` with index map lambda requirements
- Kernel function rules (Ref-based memory access, no `pl.load()`/`pl.store()`)
- TPU constraints (2D+ tensors, `(8, 128)` alignment, power-of-2 block sizes)
- A complete minimal working example (element-wise add kernel)
- Tracing/control flow rules (`jnp.where`, `pl.when`, `lax.fori_loop`)

See [`pallas_eval/prompts.py`](pallas_eval/prompts.py) for the full prompt.

---

## Reproduction

```bash
# Install dependencies
pip install openai google-generativeai python-dotenv

# Set API keys in pallas_eval/.env
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...
# TPU_IP=...

# Validate baselines
python -m pallas_eval.validate_baselines

# Generate Pallas kernels
python -m pallas_eval.generate --model gpt53
python -m pallas_eval.generate --model gemini3

# Evaluate on TPU
python -m pallas_eval.evaluate --model gpt53 --output pallas_eval/results/eval_gpt53.json
python -m pallas_eval.evaluate --model gemini3 --output pallas_eval/results/eval_gemini3.json

# Retry transient TPU init errors
python -m pallas_eval.retry_tpu_errors --input pallas_eval/results/eval_gpt53.json
python -m pallas_eval.retry_tpu_errors --input pallas_eval/results/eval_gemini3.json
```

---

## Key Takeaways

1. **1-shot Pallas generation is feasible but hard** — even frontier models only achieve 18-32% correctness on TPU Pallas code
2. **GPT-5.3 excels at complex fused operations** (45.5% on Level 2 vs 16.2% for Gemini)
3. **Gemini produces faster kernels when correct** (0.63x mean speedup vs 0.36x), with 6 kernels beating vanilla JAX
4. **The models are highly complementary** — best-of-2 reaches 42.6%, a 33% lift over GPT alone
5. **Block alignment and OOM are the dominant failure modes** — better automatic block size selection would significantly improve results
6. **Priority kernels remain the hardest** — production attention/MoE patterns are still largely out of reach for 1-shot generation
