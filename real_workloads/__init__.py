"""
Real Workloads — LLM-specific JAX baselines (one file per benchmark).

Organized by model family, each file is self-contained with CONFIG, create_inputs(),
workload(), and benchmark(). Run any file standalone to get JSON output.

Models (8 families, 36 benchmarks):
  llama3/              — GQA attention, SwiGLU MLP, RoPE, RMSNorm, Token embed (8B/70B/405B)
  llama4/              — GQA attention, RoPE, Sparse MoE top-1 (Scout 109B/Maverick 400B)
  gemma3/              — Sliding window attention, Global attention (4B/12B/27B)
  qwen3/               — GQA attention, SwiGLU MLP, Sparse MoE with shared experts (8B/14B/30B-MoE)
  mixtral/             — Sparse MoE top-2 (8x7B/8x22B)
  deepseek_v3/         — MLA attention, YaRN RoPE, MoE with shared experts (671B)
  attention_variants/  — Linear (Performer), ALiBi (BLOOM), Relative pos (T5),
                         Block-sparse (BigBird), Differential, MQA (Falcon), Cross-attn (T5)
"""

WORKLOAD_FILES = {
    "llama3": [
        "llama3_8b_gqa",
        "llama3_70b_gqa",
        "llama3_405b_gqa",
        "llama3_8b_swiglu",
        "llama3_70b_swiglu",
        "llama3_405b_swiglu",
        "llama3_8b_rope",
        "llama3_70b_rope",
        "llama3_8b_rmsnorm",
        "llama3_70b_rmsnorm",
        "llama3_8b_token_embed",
    ],
    "llama4": [
        "llama4_scout_gqa",
        "llama4_scout_moe",
        "llama4_maverick_moe",
        "llama4_scout_rope",
    ],
    "gemma3": [
        "gemma3_4b_sliding_window_attn",
        "gemma3_12b_sliding_window_attn",
        "gemma3_27b_sliding_window_attn",
        "gemma3_4b_global_attn",
        "gemma3_12b_global_attn",
        "gemma3_27b_global_attn",
    ],
    "qwen3": [
        "qwen3_8b_gqa",
        "qwen3_14b_gqa",
        "qwen3_8b_swiglu",
        "qwen3_moe_30b_moe",
    ],
    "mixtral": [
        "mixtral_8x7b_moe",
        "mixtral_8x22b_moe",
    ],
    "deepseek_v3": [
        "deepseek_v3_mla",
        "deepseek_v3_yarn_rope",
        "deepseek_v3_moe",
    ],
    "attention_variants": [
        "performer_favor_attention",
        "bloom_7b_alibi_attention",
        "t5_base_relative_attention",
        "t5_base_cross_attention",
        "bigbird_block_sparse_attention",
        "differential_attention",
        "falcon_7b_mqa_attention",
    ],
}

ALL_WORKLOADS = [w for ws in WORKLOAD_FILES.values() for w in ws]
