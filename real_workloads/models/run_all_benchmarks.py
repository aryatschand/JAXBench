#!/usr/bin/env python3
"""
Master benchmark runner for all model-specific operators.

Runs benchmarks for unique operators from each model family:
- Llama 3.1: GQA attention, SwiGLU MLP, RoPE
- Gemma 3: Sliding window attention
- Mixtral: Sparse MoE
- DeepSeek V3: MLA attention, Shared-expert MoE, YaRN RoPE
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

def run_benchmark_module(module_path: str, module_name: str):
    """Run a benchmark module and capture results."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    # Redirect stdout to capture output
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error running {module_name}: {e}")

    return output.getvalue()


def main():
    import jax

    print("=" * 100)
    print("MODEL-SPECIFIC OPERATOR BENCHMARKS")
    print("=" * 100)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Define benchmarks to run
    base_dir = Path(__file__).parent

    benchmarks = [
        # Llama 3.1
        ("Llama 3.1 GQA Attention", base_dir / "llama3" / "attention_gqa.py"),
        ("Llama 3.1 SwiGLU MLP", base_dir / "llama3" / "mlp_swiglu.py"),
        ("Llama 3.1 RoPE", base_dir / "llama3" / "rope.py"),

        # Gemma 3
        ("Gemma 3 Sliding Window", base_dir / "gemma3" / "attention_sliding_window.py"),

        # Mixtral
        ("Mixtral Sparse MoE", base_dir / "mixtral" / "moe_sparse_routing.py"),

        # DeepSeek V3
        ("DeepSeek V3 MLA", base_dir / "deepseek_v3" / "attention_mla.py"),
        ("DeepSeek V3 Shared MoE", base_dir / "deepseek_v3" / "moe_shared_experts.py"),
        ("DeepSeek V3 YaRN RoPE", base_dir / "deepseek_v3" / "rope_yarn.py"),
    ]

    results = {}
    total_start = time.time()

    for name, path in benchmarks:
        print("=" * 100)
        print(f"BENCHMARK: {name}")
        print("=" * 100)

        if not path.exists():
            print(f"  [SKIP] File not found: {path}")
            continue

        try:
            # Import and run the benchmark module
            import importlib.util
            spec = importlib.util.spec_from_file_location(name.replace(" ", "_"), path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results[name] = "success"
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[name] = f"error: {str(e)}"

        print()

    total_time = time.time() - total_start

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total benchmarks: {len(benchmarks)}")
    print(f"Successful: {sum(1 for v in results.values() if v == 'success')}")
    print(f"Failed: {sum(1 for v in results.values() if v != 'success')}")
    print(f"Total time: {total_time:.1f}s")
    print()

    for name, status in results.items():
        status_str = "[OK]" if status == "success" else "[FAIL]"
        print(f"  {status_str} {name}")


if __name__ == "__main__":
    main()
