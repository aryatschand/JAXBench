"""
Pallas kernel implementations for real workloads.

Each kernel file must define a `pallas_kernel` function that:
- Takes the same inputs as the JAX baseline
- Returns the same output shape/dtype
- Is JIT-compatible

Naming convention:
- <workload_name>.py (e.g., llama3_8b_gqa.py, llama3_8b_rope.py)
"""
