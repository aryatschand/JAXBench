# Real Workloads Benchmark

Extracted JAX operators from production LLM implementations (MaxText) with real problem sizes.

## Purpose

Unlike synthetic benchmarks, this suite contains **real operators** extracted from production LLM code:
- Actual attention implementations (Flash, Splash, Ragged)
- Real GEMM shapes from Llama, Gemma, Qwen, DeepSeek models
- Production normalization layers (RMSNorm, L2Norm)
- MoE routing and expert computation

## Source

Operators extracted from [MaxText](https://github.com/AI-Hypercomputer/maxtext) - Google's reference implementation for training LLMs on TPU.

## Structure

```
real_workloads/
├── maxtext/                    # Submodule: MaxText source
├── extracted/                  # Isolated operators ready for benchmarking
│   ├── attention/             # Attention mechanisms
│   ├── gemm/                  # Linear projections
│   ├── normalization/         # RMSNorm, L2Norm
│   ├── activations/           # GELU, SiLU, SwiGLU
│   ├── embeddings/            # RoPE, token embeddings
│   └── moe/                   # MoE components
├── benchmarks/                # Problem sizes for each model
├── scripts/                   # Extraction and benchmark scripts
└── results/                   # Benchmark results
```

## Key Operators

### Attention
- **Flash Attention**: Block-sparse masked attention
- **Splash Attention**: TPU-optimized Pallas kernels
- **Dot-Product**: Standard scaled dot-product attention

### Linear/GEMM
- **DenseGeneral**: Flexible multi-axis linear transformations
- **QKV Projections**: Fused or separate Q/K/V
- **MLP Blocks**: SwiGLU-style gated FFN

### Normalization
- **RMSNorm**: Root mean square normalization
- **L2Norm**: L2 normalization (Llama4)

### Embeddings
- **RoPE**: Rotary position embeddings (multiple variants)
- **Yarn**: Extended context RoPE
- **Token Embeddings**: Vocabulary lookup

### MoE
- **Routed MoE**: Top-k expert routing
- **Gate Logits**: Expert selection
- **Token Sorting**: Efficient expert batching

## Problem Sizes

### Llama 3.1 70B
- hidden_dim: 8192
- num_heads: 64
- head_dim: 128
- num_kv_heads: 8 (GQA)
- intermediate_dim: 28672
- seq_len: 8192

### Gemma 3 27B
- hidden_dim: 4608
- num_heads: 32
- head_dim: 144
- num_kv_heads: 8
- intermediate_dim: 36864

### DeepSeek V3
- hidden_dim: 7168
- num_experts: 256
- num_experts_per_tok: 8
- intermediate_dim: 18432

## Usage

```bash
# Extract operators from maxtext
python real_workloads/scripts/extract_operators.py

# Run benchmarks on TPU
python real_workloads/scripts/run_benchmark.py --tpu-ip <IP>

# Generate report
python real_workloads/scripts/generate_report.py
```

## Benchmark Results

See `results/` for timing and correctness data.
