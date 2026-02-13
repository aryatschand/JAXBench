"""
Model-specific JAX operators extracted from MaxText.

Each subfolder contains standalone benchmark files for unique operators
used by that model family.

Models:
- llama3: Llama 3.1 (8B/70B/405B) - GQA, SwiGLU, standard RoPE
- gemma3: Gemma 3 (4B/12B/27B) - Sliding window attention, QK norm
- mixtral: Mixtral 8x7B/8x22B - Sparse MoE with top-k routing
- deepseek_v3: DeepSeek V3 (671B) - MLA attention, shared experts MoE, YaRN RoPE
"""
