"""Inference speedup evaluation — map kernel-level optimizations to end-to-end tokens/s.

Models:
  llama3_8b  — Standard transformer (GQA + SwiGLU + RoPE + RMSNorm)
  gla_1_3b   — Gated Linear Attention transformer
  mamba2_2_7b — Mamba-2 State Space Duality model

Usage:
  python -m inference_speedup.run_eval --model llama3_8b
  python -m inference_speedup.run_eval --model gla_1_3b --optimized rmsnorm
  python -m inference_speedup.run_eval --all --profile
"""
