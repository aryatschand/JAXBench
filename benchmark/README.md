# JAXBench Benchmark Suite

50 diverse JAX/TPU kernel workloads for evaluating Pallas kernel optimization.

**17 priority kernels** from production LLM architectures (Llama, DeepSeek, Mixtral, Mamba, RetNet, AlphaFold)
**33 fused operator sequences** from KernelBench Level 2 (matmul+activation, gemm+norm, conv+pool chains)

## Quick Start

```bash
# Run a single workload
PJRT_DEVICE=TPU python3 benchmark/gemm/baseline.py

# Run Pallas-optimized variant (where available)
PJRT_DEVICE=TPU python3 benchmark/flash_attention/pallas.py
```

## Workload Format

Each workload folder contains:
- `baseline.py` — vanilla JAX implementation (the target to optimize)
- `pallas.py` — Pallas TPU kernel (where available, with autotuned block sizes)
- `optimized.py` — optimized JAX with Pallas library imports (where available)

**Priority kernels** use: `CONFIG`, `create_inputs()`, `workload()`, `benchmark()`
**Fused operators** use: `Model` class with `forward()`, `get_inputs()`, `get_init_inputs()`

## All 50 Workloads

| # | Workload | Category | Variants | Baseline (ms) | Optimized (ms) | Speedup |
|---|----------|----------|----------|----------:|----------:|--------:|
| 1 | [cross_entropy](cross_entropy/) | priority | baseline | 7.70 | — | — |
| 2 | [flash_attention](flash_attention/) | priority | baseline, pallas | 1.49 | 0.62 | 2.4x |
| 3 | [flex_attention](flex_attention/) | priority | baseline | 2.91 | — | — |
| 4 | [gemm](gemm/) | priority | baseline, pallas | 5.48 | 5.62 | 1.0x |
| 5 | [gqa_attention](gqa_attention/) | priority | baseline, optimized | 3.24 | 1.44 | 2.3x |
| 6 | [mamba2_ssd](mamba2_ssd/) | priority | baseline | 1.82 | — | — |
| 7 | [megablox_gmm](megablox_gmm/) | priority | baseline, pallas | 187.04 | 2.82 | 66.2x |
| 8 | [mla_attention](mla_attention/) | priority | baseline, optimized | 4.48 | 4.28 | 1.0x |
| 9 | [paged_attention](paged_attention/) | priority | baseline, pallas | 1.96 | 1.02 | 1.9x |
| 10 | [ragged_dot](ragged_dot/) | priority | baseline | 1.37 | — | — |
| 11 | [ragged_paged_attention](ragged_paged_attention/) | priority | baseline, pallas | 191.99 | 0.86 | 222.0x |
| 12 | [retnet_retention](retnet_retention/) | priority | baseline | 0.52 | — | — |
| 13 | [rms_norm](rms_norm/) | priority | baseline | 0.17 | — | — |
| 14 | [sparse_attention](sparse_attention/) | priority | baseline, pallas | 1.50 | 0.66 | 2.3x |
| 15 | [sparse_moe](sparse_moe/) | priority | baseline | 8.29 | — | — |
| 16 | [swiglu_mlp](swiglu_mlp/) | priority | baseline | 4.07 | — | — |
| 17 | [triangle_multiplication](triangle_multiplication/) | priority | baseline | 1.31 | — | — |
| 18 | [12_Gemm_Multiply_LeakyReLU](12_Gemm_Multiply_LeakyReLU/) | gemm fused | baseline | — | — | — |
| 19 | [14_Gemm_Divide_Sum_Scaling](14_Gemm_Divide_Sum_Scaling/) | gemm fused | baseline | — | — | — |
| 20 | [17_Conv2d_InstanceNorm_Divide](17_Conv2d_InstanceNorm_Divide/) | conv2d fused | baseline | — | — | — |
| 21 | [18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp](18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/) | matmul fused | baseline | — | — | — |
| 22 | [1_Conv2D_ReLU_BiasAdd](1_Conv2D_ReLU_BiasAdd/) | conv2d fused | baseline | — | — | — |
| 23 | [22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish](22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish/) | matmul fused | baseline | — | — | — |
| 24 | [23_Conv3d_GroupNorm_Mean](23_Conv3d_GroupNorm_Mean/) | conv3d fused | baseline | — | — | — |
| 25 | [28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply](28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply/) | bmm fused | baseline | — | — | — |
| 26 | [29_Matmul_Mish_Mish](29_Matmul_Mish_Mish/) | matmul fused | baseline | — | — | — |
| 27 | [34_ConvTranspose3d_LayerNorm_GELU_Scaling](34_ConvTranspose3d_LayerNorm_GELU_Scaling/) | convtranspose fused | baseline | — | — | — |
| 28 | [37_Matmul_Swish_Sum_GroupNorm](37_Matmul_Swish_Sum_GroupNorm/) | matmul fused | baseline | — | — | — |
| 29 | [40_Matmul_Scaling_ResidualAdd](40_Matmul_Scaling_ResidualAdd/) | matmul fused | baseline | — | — | — |
| 30 | [41_Gemm_BatchNorm_GELU_ReLU](41_Gemm_BatchNorm_GELU_ReLU/) | gemm fused | baseline | — | — | — |
| 31 | [45_Gemm_Sigmoid_LogSumExp](45_Gemm_Sigmoid_LogSumExp/) | gemm fused | baseline | — | — | — |
| 32 | [47_Conv3d_Mish_Tanh](47_Conv3d_Mish_Tanh/) | conv3d fused | baseline | — | — | — |
| 33 | [52_Conv2d_Activation_BatchNorm](52_Conv2d_Activation_BatchNorm/) | conv2d fused | baseline | — | — | — |
| 34 | [53_Gemm_Scaling_Hardtanh_GELU](53_Gemm_Scaling_Hardtanh_GELU/) | gemm fused | baseline | — | — | — |
| 35 | [56_Matmul_Sigmoid_Sum](56_Matmul_Sigmoid_Sum/) | matmul fused | baseline | — | — | — |
| 36 | [59_Matmul_Swish_Scaling](59_Matmul_Swish_Scaling/) | matmul fused | baseline | — | — | — |
| 37 | [66_Matmul_Dropout_Softmax](66_Matmul_Dropout_Softmax/) | matmul fused | baseline | — | — | — |
| 38 | [67_Conv2d_GELU_GlobalAvgPool](67_Conv2d_GELU_GlobalAvgPool/) | conv2d fused | baseline | — | — | — |
| 39 | [75_Gemm_GroupNorm_Min_BiasAdd](75_Gemm_GroupNorm_Min_BiasAdd/) | gemm fused | baseline | — | — | — |
| 40 | [76_Gemm_Add_ReLU](76_Gemm_Add_ReLU/) | gemm fused | baseline | — | — | — |
| 41 | [80_Gemm_Max_Subtract_GELU](80_Gemm_Max_Subtract_GELU/) | gemm fused | baseline | — | — | — |
| 42 | [84_Gemm_BatchNorm_Scaling_Softmax](84_Gemm_BatchNorm_Scaling_Softmax/) | gemm fused | baseline | — | — | — |
| 43 | [86_Matmul_Divide_GELU](86_Matmul_Divide_GELU/) | matmul fused | baseline | — | — | — |
| 44 | [88_Gemm_GroupNorm_Swish_Multiply_Swish](88_Gemm_GroupNorm_Swish_Multiply_Swish/) | gemm fused | baseline | — | — | — |
| 45 | [92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp](92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp/) | conv2d fused | baseline | — | — | — |
| 46 | [95_Matmul_Add_Swish_Tanh_GELU_Hardtanh](95_Matmul_Add_Swish_Tanh_GELU_Hardtanh/) | matmul fused | baseline | — | — | — |
| 47 | [97_Matmul_BatchNorm_BiasAdd_Divide_Swish](97_Matmul_BatchNorm_BiasAdd_Divide_Swish/) | matmul fused | baseline | — | — | — |
| 48 | [98_Matmul_AvgPool_GELU_Scale_Max](98_Matmul_AvgPool_GELU_Scale_Max/) | matmul fused | baseline | — | — | — |
| 49 | [99_Matmul_GELU_Softmax](99_Matmul_GELU_Softmax/) | matmul fused | baseline | — | — | — |
| 50 | [9_Matmul_Subtract_Multiply_ReLU](9_Matmul_Subtract_Multiply_ReLU/) | matmul fused | baseline | — | — | — |

## Category Breakdown

| Category | Count |
|----------|------:|
| priority kernel | 17 |
| matmul fused | 14 |
| gemm fused | 10 |
| conv2d fused | 5 |
| conv3d fused | 2 |
| bmm fused | 1 |
| convtranspose fused | 1 |
