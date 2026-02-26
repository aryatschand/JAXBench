"""
Tokamax Workloads — TPU kernel benchmarks from openxla/tokamax.

6 operation categories, 12 workloads total. Each file is self-contained with
CONFIG, create_inputs(), workload(), and benchmark(). Run any file standalone
to get JSON output.

Operations:
  attention/          — Scaled dot-product attention with GQA + causal masking (2 workloads)
  ragged_dot/         — Grouped matmul for MoE routing (1 workload)
  cross_entropy/      — Fused linear projection + softmax + cross-entropy loss (3 workloads)
  gated_linear_unit/  — SwiGLU activation (2 workloads)
  layer_norm/         — RMSNorm (2 workloads)
  triangle_mult/      — AlphaFold triangle multiplicative update (2 workloads)
"""

WORKLOAD_FILES = {
    "attention": [
        "mixtral_8x7b_attention",
        "deepseek2_16b_attention",
    ],
    "ragged_dot": [
        "mixtral_8x7b_ragged_dot",
    ],
    "cross_entropy": [
        "llama3_8b_cross_entropy",
        "qwen3_8b_cross_entropy",
        "gemma3_4b_cross_entropy",
    ],
    "gated_linear_unit": [
        "llama3_8b_swiglu",
        "llama3_70b_swiglu",
    ],
    "layer_norm": [
        "llama3_8b_rmsnorm",
        "llama3_70b_rmsnorm",
    ],
    "triangle_mult": [
        "alphafold_384_triangle_mult",
        "alphafold_768_triangle_mult",
    ],
}

ALL_WORKLOADS = [w for ws in WORKLOAD_FILES.values() for w in ws]
